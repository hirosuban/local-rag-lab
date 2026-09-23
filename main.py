import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# 記事スタイル: 多言語 e5 モデル + passage:/query: プレフィックス + 正規化済みベクトル
# ※ e5 系は文書に "passage: "、クエリに "query: " を付けるのが必須
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

print(f"Embedding モデルを読み込み中... ({EMBED_MODEL_NAME})")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# 意図的に似た単語（手当・リモート・申請・費用など）を散りばめたナレッジベース
knowledge_base = [
    # 紛らわしい手当・費用関連
    "通勤手当は月額上限30,000円まで実費支給されます（定期券代ベースで計算）。",

    # 紛らわしいリモート・作業環境関連
    "リモートワーク用のPC周辺機器（モニター、キーボード等）の購入費用は、年間20,000円まで経費精算が可能です。",
    "コワーキングスペース利用料は、月額10,000円を上限として事前申請の上で実費精算が認められます。",
    "フルリモート契約社員の光熱費補助制度は2025年3月をもって廃止され、基本給へ統合されました。",

    # 正解データ（本命）
    "一般社員のリモートワーク手当（通信費・光熱費の補助）は月額15,000円で、毎月の給与と合算して支給されます。",

    # 紛らわしい申請・ルール関連
    "有給休暇の取得申請は、希望日の3営業日前までに社内ポータルから提出してください。",
    "健康診断の再検査費用は、上限10,000円まで会社が負担します（領収書の提出が必要）。",
]

print(f"ナレッジベース件数: {len(knowledge_base)} 件")

# 文書側は "passage: " を付けてエンコード。normalize_embeddings=True で単位ベクトル化。
doc_embeddings = embed_model.encode(
    [f"passage: {doc}" for doc in knowledge_base],
    normalize_embeddings=True,
)
print(f"ベクトルの形状: {doc_embeddings.shape}")

# 曖昧な口語の質問。クエリ側は "query: " を付ける。
query = "家で仕事するときの手当ってもらえるの？いくら？"
query_embedding = embed_model.encode(f"query: {query}", normalize_embeddings=True)

# 正規化済みベクトルなので、内積 = コサイン類似度。行列演算で全件を一発計算。
scores = doc_embeddings @ query_embedding

# 上位3件を表示して、迷った候補を確認
top3_indices = np.argsort(scores)[::-1][:3]

print(f"\n[質問]: {query}")
print("--- [類似度スコア 上位3件] ---")
for rank, idx in enumerate(top3_indices, 1):
    print(f"{rank}位 (スコア: {scores[idx]:.4f}): {knowledge_base[idx]}")
print("------------------------------\n")

# 検索で拾った上位3件をすべて LLM に渡す。
# top1 だけだと正解が2〜3位のとき取りこぼす（記事の④再ランキング・文脈構成の失敗）。
context = "\n".join(
    f"- {knowledge_base[idx]}" for idx in top3_indices
)

prompt = f"""
以下の[参考情報]だけをもとにして、[質問]に日本語で簡潔に回答してください。
参考情報に書かれていないことは絶対に回答に含めないでください。

[参考情報]
{context}

[質問]
{query}
"""

print("Ollama で回答を生成中...")
response = ollama.chat(
    model="qwen2.5:0.5b",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.0},
)

print("[LLMの回答]:")
print(response["message"]["content"])
