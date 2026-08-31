# local-rag-lab

ローカルだけで動く RAG（ベクトル検索 → LLM 生成）の実験場。
依存インストールは devcontainer 起動時に実行される。

- `main.py` … numpy で全件総当たり検索
- `main_duckdb.py` … DuckDB VSS に永続化 + HNSW インデックスで検索

## 動作手順

```bash
# 1. Ollama サーバーを起動（起動済みなら不要）
ollama serve &

# 2. LLM モデルを取得（初回のみ）
ollama pull qwen2.5:0.5b

# 3. 実行
python main.py          # または python main_duckdb.py
```