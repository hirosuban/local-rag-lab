import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

print("日本語対応Embeddingモデルを読み込み中...")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 意図的に似た単語（手当・リモート・申請・費用など）を散りばめた分厚いナレッジベース
knowledge_base = [
    # 紛らわしい手当・費用関連
    "通勤手当は月額上限30,000円まで実費支給されます（定期券代ベースで計算）。",
    "役職手当はリーダー職が月額20,000円、マネージャー職が月額50,000円です。",
    "深夜勤務手当は基本給の25%増しとして、該当時間の勤怠打刻から自動計算されます。",
    "資格取得支援制度により、会社指定のIT資格に合格した際は受験料全額と一時金50,000円が支給されます。",
    "出張時の日当は国内出張が1日あたり2,000円、海外出張が1日あたり5,000円です。",
    
    # 紛らわしいリモート・作業環境関連
    "リモートワーク用のPC周辺機器（モニター、キーボード等）の購入費用は、年間20,000円まで経費精算が可能です。",
    "在宅勤務時のセキュリティガイドラインとして、カフェ等での公共Wi-Fiへの直接接続は禁止されています。",
    "リモートワークを実施する日は、始業時と終業時に必ずSlackの #general チャンネルで報告してください。",
    "コワーキングスペース利用料は、月額10,000円を上限として事前申請の上で実費精算が認められます。",
    "フルリモート契約社員の光熱費補助制度は2025年3月をもって廃止され、基本給へ統合されました。",

    # 正解データ（本命）
    "一般社員のリモートワーク手当（通信費・光熱費の補助）は月額15,000円で、毎月の給与と合算して支給されます。",

    # 紛らわしいプロジェクト・予算関連
    "プロジェクト・フェニックスの予算上限は850万円で、期間は2026年12月末までです。",
    "新規事業開発PJのプロモーション費用は、第3四半期のみ月額50万円が別途配分されます。",
    "外部業務委託メンバーへの支払いは、月末締め翌月末払いで処理されます。",

    # 紛らわしい申請・ルール関連
    "有給休暇の取得申請は、希望日の3営業日前までに社内ポータルから提出してください。",
    "慶弔休暇および慶弔見舞金の申請は、事由発生から1ヶ月以内に人事部へ証明書類を提出してください。",
    "定期サーバーメンテナンスは毎週火曜日の午前11:00（JST）から30分間実施されます。",
    "社内PCのパスワードは90日ごとに変更が必要で、過去3世代と同じパスワードは設定できません。",
    "オフィス内での私用電話は、3階のリフレッシュルームまたは各階の通話ブースを利用してください。",
    "健康診断の再検査費用は、上限10,000円まで会社が負担します（領収書の提出が必要）。"
]

print(f"ナレッジベース件数: {len(knowledge_base)} 件")
doc_embeddings = embed_model.encode(knowledge_base)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print(doc_embeddings)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 曖昧な口語の質問
query = "家で仕事するときの手当ってもらえるの？いくら？"
query_embedding = embed_model.encode(query)

# 全件の類似度スコアを計算
scores = np.array([cosine_similarity(query_embedding, emb) for emb in doc_embeddings])

# 上位3件を表示して、迷った候補を確認
top3_indices = np.argsort(scores)[::-1][:3]

print(f"\n[質問]: {query}")
print("--- [類似度スコア 上位3件] ---")
for rank, idx in enumerate(top3_indices, 1):
    print(f"{rank}位 (スコア: {scores[idx]:.4f}): {knowledge_base[idx]}")
print("------------------------------\n")

best_doc = knowledge_base[top3_indices[0]]

prompt = f"""
以下の[参考情報]だけをもとにして、[質問]に日本語で簡潔に回答してください。
参考情報に書かれていないことは絶対に回答に含めないでください。

[参考情報]
{best_doc}

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
