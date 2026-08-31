"""DuckDB VSS 版: ベクトルを DuckDB に永続化し、HNSW インデックスで近傍検索する。

main.py は numpy で全件総当たり。こちらは記事の DuckDB VSS の書き方に置き換えた版。
- embedding を FLOAT[768] カラムに保存
- HNSW インデックス + array_cosine_distance() で検索
- WHERE 句での絞り込みと類似度検索を 1 クエリで書ける
"""

import duckdb
import ollama
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIM = 768
DB_PATH = "vectors.duckdb"

# 意図的に似た単語（手当・リモート・申請・費用など）を散りばめたナレッジベース
knowledge_base = [
    "通勤手当は月額上限30,000円まで実費支給されます（定期券代ベースで計算）。",
    "リモートワーク用のPC周辺機器（モニター、キーボード等）の購入費用は、年間20,000円まで経費精算が可能です。",
    "コワーキングスペース利用料は、月額10,000円を上限として事前申請の上で実費精算が認められます。",
    "フルリモート契約社員の光熱費補助制度は2025年3月をもって廃止され、基本給へ統合されました。",
    "一般社員のリモートワーク手当（通信費・光熱費の補助）は月額15,000円で、毎月の給与と合算して支給されます。",
    "有給休暇の取得申請は、希望日の3営業日前までに社内ポータルから提出してください。",
    "健康診断の再検査費用は、上限10,000円まで会社が負担します（領収書の提出が必要）。",
]

print(f"Embedding モデルを読み込み中... ({EMBED_MODEL_NAME})")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# --- DuckDB 準備 -----------------------------------------------------------
con = duckdb.connect(DB_PATH)
con.execute("INSTALL vss; LOAD vss;")
# 永続 DB に HNSW インデックスを作るには実験フラグが必要
con.execute("SET hnsw_enable_experimental_persistence = true")

con.execute("DROP TABLE IF EXISTS documents")
con.execute(f"""
    CREATE TABLE documents (
        id INTEGER,
        content TEXT,
        embedding FLOAT[{EMBED_DIM}]
    )
""")

# --- Embedding 生成 & 登録 ------------------------------------------------
# e5 系は文書に "passage: " を付ける。normalize_embeddings=True で単位ベクトル化。
doc_embeddings = embed_model.encode(
    [f"passage: {doc}" for doc in knowledge_base],
    normalize_embeddings=True,
)

for i, (doc, emb) in enumerate(zip(knowledge_base, doc_embeddings)):
    con.execute(
        "INSERT INTO documents VALUES (?, ?, ?)",
        [i, doc, emb.tolist()],
    )

count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
print(f"登録件数: {count} 件")

# HNSW インデックス（コサイン距離）
con.execute("""
    CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING HNSW (embedding) WITH (metric = 'cosine')
""")

# --- 検索 ----------------------------------------------------------------
query = "家で仕事するときの手当ってもらえるの？いくら？"
query_embedding = embed_model.encode(
    f"query: {query}", normalize_embeddings=True
).tolist()

results = con.execute(f"""
    SELECT content, array_cosine_distance(embedding, ?::FLOAT[{EMBED_DIM}]) AS distance
    FROM documents
    ORDER BY distance ASC
    LIMIT 3
""", [query_embedding]).fetchall()

print(f"\n[質問]: {query}")
print("--- [類似度スコア 上位3件] ---")
for rank, (content, distance) in enumerate(results, 1):
    similarity = 1 - distance  # コサイン距離 → コサイン類似度
    print(f"{rank}位 (スコア: {similarity:.4f}): {content}")
print("------------------------------\n")

con.close()

best_doc = results[0][0]

# --- 生成 (Ollama) -----------------------------------------------------
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
