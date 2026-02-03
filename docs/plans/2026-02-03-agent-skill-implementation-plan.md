# Agent-Skill 投标评分系统 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在本地落地基于 MCP Server + Postgres 的投标评分系统最小可用闭环（离线索引、在线检索、证据评分、引用验证、评测回归）。

**Architecture:** Python 实现数据入库与检索逻辑，MCP Server 提供只读工具接口，Skills 负责评分与验证；评测用离线数据集与回归测试闭环。

**Tech Stack:** Python 3.11, Postgres 15+, pgvector, psycopg, openai, python-dotenv, FastMCP(或等价 MCP SDK), pytest, deepeval

**Note:** 若当前目录不是 git 仓库，先执行 `git init`，否则跳过所有 "Step 5: Commit" 步骤。

---

### 规划要求（AGENTS 对齐）
- 任务拆解：每个任务控制在 2-5 分钟的单一动作，避免跨职责改动。  
- Under-Prompt：仅定义目标与约束，不预设实现细节；允许在执行中最小化改动与回退。  
- 自治标准：执行每个任务前，必须补齐“输入、输出、依赖、失败与回退”的最小说明。  

### 范围与非目标
- 范围：本地最小闭环（入库、检索、评分、引用验证、评测回归）。  
- 非目标：生产级高可用、权限体系、分布式部署与多租户隔离。  

### 关键假设
- 本地可用 Postgres 15+ 且可安装 pgvector。  
- 评测所需网络与 `OPENAI_API_KEY` 可用，否则评测跳过。  
- MCP SDK 选型在实现阶段确定，必要时替换 `fastmcp`。  

### 执行前自治检查（每个任务都要满足）
- 输入：明确依赖的环境变量/参数/前置数据。  
- 输出：明确产出文件与可验证的行为。  
- 依赖：注明跨任务或跨模块的依赖与顺序。  
- 失败与回退：失败时的诊断点与回退方式。  

---

### 最佳实践补充（基于官方文档）
- 全文检索：`tsvector` 列使用 GIN 索引，适合高频检索场景。  
- 向量索引选型：HNSW 召回/速度表现更佳但构建慢、占内存；IVFFlat 构建更快、内存更省但召回略低。  
- 向量检索调参：查询时可按索引类型设置 `SET LOCAL hnsw.ef_search = ...` 或 `SET LOCAL ivfflat.probes = ...`。  
- 索引构建：生产环境可用 `CREATE INDEX CONCURRENTLY`，必要时提高 `maintenance_work_mem` 与并行维护参数加速建索引。  
- psycopg3 事务：连接上下文会自动提交/回滚，脚本需确保事务最终提交或回滚。  
- OpenAI 配置：API Key 与模型配置建议通过环境变量注入，避免硬编码到代码库。  
- OpenAI 客户端支持 `base_url` 用于代理/自定义端点。  
- OpenAI 客户端支持 `timeout` 与 `max_retries`，并可通过环境变量配置。  
- 本地开发推荐使用 `.env`（`python-dotenv`）加载配置，部署环境以系统环境变量为准。  
- LLM 任务建议按用途拆分模型，通过 `OPENAI_LLM_MODEL_<TASK>` 实现灵活切换。  
- 结构化输出推荐使用 `response_format: json_schema` 且 `strict: true`，优于仅保证 JSON 合法性的 json_object。  

---

### Task 1: 初始化目录与依赖

**自治检查（Task 1）**
- 输入：Python 3.11、pip；无数据库依赖。  
- 输出：`requirements.txt`、`bid_scoring/config.py`、`.env.example`、`tests/test_config.py`；`pytest tests/test_config.py -q` 通过。  
- 依赖：无。  
- 失败与回退：删除新增文件并移除依赖后重试；若测试失败，检查环境变量读取逻辑。  

**Files:**
- Create: `requirements.txt`
- Create: `bid_scoring/__init__.py`
- Create: `bid_scoring/config.py`
- Create: `.env.example`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**
```python
# tests/test_config.py
from bid_scoring.config import load_settings

def test_load_settings_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_LLM_MODEL_DEFAULT", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_LLM_MODEL_SCORING", "gpt-4o")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIM", "1536")
    settings = load_settings()
    assert settings["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert settings["OPENAI_LLM_MODEL_DEFAULT"] == "gpt-4o-mini"
    assert settings["OPENAI_LLM_MODELS"]["scoring"] == "gpt-4o"
    assert settings["OPENAI_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert settings["OPENAI_EMBEDDING_DIM"] == 1536
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_config.py -q`  
Expected: FAIL with "ModuleNotFoundError: No module named 'bid_scoring'"

**Step 3: Write minimal implementation**
```python
# bid_scoring/__init__.py
__all__ = ["config"]
```

```python
# bid_scoring/config.py
import os
from dotenv import load_dotenv

def _load_task_models(prefix: str, default_key: str) -> dict:
    models: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix) and key != default_key and value:
            task = key[len(prefix):].lower()
            models[task] = value
    return models

def load_settings() -> dict:
    load_dotenv()
    default_llm = os.getenv("OPENAI_LLM_MODEL_DEFAULT") or os.getenv("OPENAI_LLM_MODEL")
    return {
        "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://localhost:5432/bid_scoring"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "OPENAI_LLM_MODEL_DEFAULT": default_llm,
        "OPENAI_LLM_MODELS": _load_task_models("OPENAI_LLM_MODEL_", "OPENAI_LLM_MODEL_DEFAULT"),
        "OPENAI_TIMEOUT": float(os.getenv("OPENAI_TIMEOUT", "0") or 0),
        "OPENAI_MAX_RETRIES": int(os.getenv("OPENAI_MAX_RETRIES", "0") or 0),
        "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL"),
        "OPENAI_EMBEDDING_DIM": int(os.getenv("OPENAI_EMBEDDING_DIM", "0")),
    }
```

```txt
# requirements.txt
psycopg[binary]
pgvector
openai
python-dotenv
fastmcp
pytest
deepeval
```

```env
# .env.example
DATABASE_URL=postgresql://localhost:5432/bid_scoring
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_LLM_MODEL_DEFAULT=gpt-4o-mini
OPENAI_LLM_MODEL_SCORING=gpt-4o
OPENAI_LLM_MODEL_VERIFY=gpt-4o-mini
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=2
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIM=1536
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_config.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add .env.example requirements.txt bid_scoring/__init__.py bid_scoring/config.py tests/test_config.py
git commit -m "chore(core): init config module"
```

---

### Task 2: 建库与基础表结构

**自治检查（Task 2）**
- 输入：Postgres 15+ 已启动；`DATABASE_URL` 可连接；`OPENAI_EMBEDDING_DIM` 已设置。  
- 输出：`migrations/001_init.sql`、`scripts/apply_migrations.py`、`tests/test_db_schema.py`；库表与索引创建成功。  
- 依赖：Task 1（配置加载）。  
- 失败与回退：失败时检查扩展 `pgcrypto`/`vector`；回退可删表或重建本地数据库。  

**Files:**
- Create: `migrations/001_init.sql`
- Create: `scripts/apply_migrations.py`
- Create: `tests/test_db_schema.py`

**Step 1: Write the failing test**
```python
# tests/test_db_schema.py
import psycopg
from bid_scoring.config import load_settings

def test_tables_exist():
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from information_schema.tables where table_name='projects'")
            assert cur.fetchone()[0] == 1
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_db_schema.py -q`  
Expected: FAIL with "relation 'projects' does not exist"

**Step 3: Write minimal implementation**
```sql
-- migrations/001_init.sql
create extension if not exists pgcrypto;
create extension if not exists vector;

create table if not exists projects (
  project_id uuid primary key,
  name text not null,
  owner text,
  status text,
  created_at timestamptz default now()
);

create table if not exists documents (
  doc_id uuid primary key,
  project_id uuid references projects(project_id),
  title text,
  source_type text,
  created_at timestamptz default now()
);

create table if not exists document_versions (
  version_id uuid primary key,
  doc_id uuid references documents(doc_id),
  source_uri text,
  source_hash text,
  parser_version text,
  created_at timestamptz default now(),
  status text
);

create table if not exists chunks (
  chunk_id uuid primary key,
  project_id uuid references projects(project_id),
  version_id uuid references document_versions(version_id),
  source_id text,
  chunk_index int,
  page_idx int,
  bbox jsonb,
  element_type text,
  text_raw text,
  text_hash text,
  text_tsv tsvector,
  embedding vector({{EMBEDDING_DIM}})
);

create table if not exists scoring_runs (
  run_id uuid primary key,
  project_id uuid references projects(project_id),
  version_id uuid references document_versions(version_id),
  dimensions text[],
  model text,
  rules_version text,
  params_hash text,
  started_at timestamptz default now(),
  status text
);

create table if not exists scoring_results (
  result_id uuid primary key,
  run_id uuid references scoring_runs(run_id),
  dimension text,
  score numeric,
  max_score numeric,
  reasoning text,
  evidence_found boolean,
  confidence text
);

create table if not exists citations (
  citation_id uuid primary key,
  result_id uuid references scoring_results(result_id),
  source_id text,
  chunk_id uuid references chunks(chunk_id),
  cited_text text,
  verified boolean,
  match_type text
);

create index if not exists idx_chunks_text_tsv on chunks using gin(text_tsv);
-- 向量索引（择一）：HNSW 召回更好但更吃内存
create index if not exists idx_chunks_embedding_hnsw on chunks using hnsw(embedding vector_cosine_ops);
-- 向量索引（可选）：IVFFlat 构建更快、内存更省，但需调参
-- create index if not exists idx_chunks_embedding_ivfflat on chunks using ivfflat(embedding vector_cosine_ops) with (lists = 100);
create index if not exists idx_chunks_project_version_page on chunks(project_id, version_id, page_idx);
```

```python
# scripts/apply_migrations.py
import pathlib
import psycopg
from bid_scoring.config import load_settings

def main():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]
    dim = settings["OPENAI_EMBEDDING_DIM"]
    if not dim:
        raise ValueError("OPENAI_EMBEDDING_DIM is required")
    template = pathlib.Path("migrations/001_init.sql").read_text(encoding="utf-8")
    sql = template.replace("{{EMBEDDING_DIM}}", str(dim))
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()

if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**
Run: `python scripts/apply_migrations.py && pytest tests/test_db_schema.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add migrations/001_init.sql scripts/apply_migrations.py tests/test_db_schema.py
git commit -m "chore(db): add initial schema"
```

---

### Task 3: MineRU 解析产物入库

**自治检查（Task 3）**
- 输入：Task 2 已完成；`tests/fixtures/sample_content_list.json` 可读；有效的 `project_id`/`document_id`/`version_id`。  
- 输出：`bid_scoring/ingest.py`、`scripts/ingest_mineru.py`、`tests/test_ingest.py`；`chunks`/`documents`/`document_versions` 有记录。  
- 依赖：Task 1-2（配置与库表）。  
- 失败与回退：失败时检查外键与输入 JSON；回退可按 `project_id`/`version_id` 删除插入记录。  

**Files:**
- Create: `bid_scoring/ingest.py`
- Create: `scripts/ingest_mineru.py`
- Create: `tests/fixtures/sample_content_list.json`
- Create: `tests/test_ingest.py`

**Step 1: Write the failing test**
```python
# tests/test_ingest.py
import json
from pathlib import Path
import psycopg
from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list

def test_ingest_inserts_chunks():
    data = json.loads(Path("tests/fixtures/sample_content_list.json").read_text(encoding="utf-8"))
    dsn = load_settings()["DATABASE_URL"]
    project_id = "00000000-0000-0000-0000-000000000001"
    document_id = "00000000-0000-0000-0000-000000000003"
    version_id = "00000000-0000-0000-0000-000000000002"
    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档",
            source_type="mineru",
            content_list=data,
        )
        with conn.cursor() as cur:
            cur.execute("select count(*) from documents where doc_id = %s", (document_id,))
            assert cur.fetchone()[0] == 1
            cur.execute("select count(*) from document_versions where version_id = %s", (version_id,))
            assert cur.fetchone()[0] == 1
            cur.execute("select count(*) from chunks")
            assert cur.fetchone()[0] > 0
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_ingest.py -q`  
Expected: FAIL with "ImportError: cannot import name 'ingest_content_list'"

**Step 3: Write minimal implementation**
```python
# bid_scoring/ingest.py
import hashlib

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def ingest_content_list(
    conn,
    project_id: str,
    document_id: str,
    version_id: str,
    content_list: list[dict],
    document_title: str = "untitled",
    source_type: str = "mineru",
    source_uri: str | None = None,
    parser_version: str | None = None,
    status: str = "ready",
) -> None:
    rows = []
    for i, item in enumerate(content_list):
        if item.get("type") not in ["text", "table"]:
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        rows.append((
            project_id,
            version_id,
            f"chunk_{i:04d}",
            i,
            item.get("page_idx", 0),
            item.get("bbox", []),
            item.get("type"),
            text,
            _hash_text(text),
        ))
    with conn.cursor() as cur:
        cur.execute(
            "insert into projects (project_id, name) values (%s, %s) on conflict do nothing",
            (project_id, f"project-{project_id[:8]}"),
        )
        cur.execute(
            "insert into documents (doc_id, project_id, title, source_type) values (%s, %s, %s, %s) on conflict do nothing",
            (document_id, project_id, document_title, source_type),
        )
        cur.execute(
            """
            insert into document_versions (version_id, doc_id, source_uri, source_hash, parser_version, status)
            values (%s, %s, %s, %s, %s, %s)
            on conflict do nothing
            """,
            (version_id, document_id, source_uri, None, parser_version, status),
        )
        cur.executemany(
            """
            insert into chunks (
              chunk_id, project_id, version_id, source_id, chunk_index, page_idx,
              bbox, element_type, text_raw, text_hash, text_tsv
            )
            values (gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
            """,
            [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[7]) for r in rows],
        )
    conn.commit()
```

```python
# scripts/ingest_mineru.py
import json
import psycopg
import argparse
from pathlib import Path
from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--document-title", default="untitled")
    parser.add_argument("--source-type", default="mineru")
    parser.add_argument("--source-uri")
    parser.add_argument("--parser-version")
    parser.add_argument("--status", default="ready")
    args = parser.parse_args()
    dsn = load_settings()["DATABASE_URL"]
    path = Path(args.path)
    content_list = json.loads(path.read_text(encoding="utf-8"))
    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            args.project_id,
            args.document_id,
            args.version_id,
            content_list,
            document_title=args.document_title,
            source_type=args.source_type,
            source_uri=args.source_uri,
            parser_version=args.parser_version,
            status=args.status,
        )

if __name__ == "__main__":
    main()
```

```json
// tests/fixtures/sample_content_list.json
[
  {"type": "text", "text": "培训时间：2天，含安装培训、操作培训", "page_idx": 12, "bbox": [100, 200, 300, 240]},
  {"type": "text", "text": "提供5年质保服务，含现场支持", "page_idx": 34, "bbox": [120, 260, 360, 300]}
]
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_ingest.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add bid_scoring/ingest.py scripts/ingest_mineru.py tests/fixtures/sample_content_list.json tests/test_ingest.py
git commit -m "feat(ingest): add mineru ingest"
```

---

### Task 4: 向量生成与更新

**自治检查（Task 4）**
- 输入：Task 3 已完成且 `chunks` 有文本；`OPENAI_API_KEY`/`OPENAI_EMBEDDING_MODEL`/`OPENAI_EMBEDDING_DIM` 可用。  
- 输出：`bid_scoring/embeddings.py`、`scripts/build_embeddings.py`、`tests/test_embeddings.py`；`chunks.embedding` 更新。  
- 依赖：Task 1-3。  
- 失败与回退：失败时检查模型与网络；回退可将已更新 `embedding` 置空再重跑。  

**Files:**
- Create: `bid_scoring/embeddings.py`
- Create: `scripts/build_embeddings.py`
- Create: `tests/test_embeddings.py`

**Step 1: Write the failing test**
```python
# tests/test_embeddings.py
from types import SimpleNamespace
from bid_scoring.embeddings import embed_texts
from bid_scoring.config import load_settings

class FakeClient:
    def __init__(self, dim: int):
        self.dim = dim
        self.embeddings = self

    def create(self, model, input):
        data = [SimpleNamespace(embedding=[0.0] * self.dim) for _ in input]
        return SimpleNamespace(data=data)

def test_embed_texts_uses_client(monkeypatch):
    monkeypatch.setenv("OPENAI_EMBEDDING_DIM", "1536")
    dim = load_settings()["OPENAI_EMBEDDING_DIM"]
    vecs = embed_texts(["a", "b"], client=FakeClient(dim), model="text-embedding-3-small")
    assert len(vecs) == 2
    assert len(vecs[0]) == dim
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_embeddings.py -q`  
Expected: FAIL with "ImportError: cannot import name 'embed_texts'"

**Step 3: Write minimal implementation**
```python
# bid_scoring/embeddings.py
from openai import OpenAI
from bid_scoring.config import load_settings

def embed_texts(texts: list[str], client: OpenAI | None = None, model: str | None = None):
    settings = load_settings()
    if client is None:
        timeout = settings["OPENAI_TIMEOUT"] or None
        max_retries = settings["OPENAI_MAX_RETRIES"] or None
        client = OpenAI(
            api_key=settings["OPENAI_API_KEY"],
            base_url=settings["OPENAI_BASE_URL"],
            timeout=timeout,
            max_retries=max_retries,
        )
    if model is None:
        model = settings["OPENAI_EMBEDDING_MODEL"]
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]
```

```python
# scripts/build_embeddings.py
import psycopg
from pgvector.psycopg import register_vector
from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_texts

def main():
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("select chunk_id, text_raw from chunks where embedding is null limit 200")
            rows = cur.fetchall()
        if not rows:
            return
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        vecs = embed_texts(texts)
        with conn.cursor() as cur:
            cur.executemany(
                "update chunks set embedding = %s where chunk_id = %s",
                [(v, i) for v, i in zip(vecs, ids)]
            )
        conn.commit()

if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_embeddings.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add bid_scoring/embeddings.py scripts/build_embeddings.py tests/test_embeddings.py
git commit -m "feat(embeddings): add embedding pipeline"
```

---

### Task 5: 检索与 RRF 融合

**自治检查（Task 5）**
- 输入：Task 4 已完成；`chunks` 含 embedding 与 tsvector 索引。  
- 输出：`bid_scoring/search.py`、`tests/test_rrf.py`；RRF 单测通过。  
- 依赖：Task 2、Task 4。  
- 失败与回退：失败时检查融合权重与输入格式；回退可删新增文件。  

**Files:**
- Create: `bid_scoring/search.py`
- Create: `tests/test_rrf.py`

**Step 1: Write the failing test**
```python
# tests/test_rrf.py
from bid_scoring.search import rrf_fuse

def test_rrf_fuse_prefers_top_ranks():
    bm25 = [("a", 1), ("b", 2), ("c", 3)]
    vec = [("c", 1), ("a", 2), ("d", 3)]
    fused = rrf_fuse(bm25, vec, k=60, bm25_weight=0.4, vector_weight=0.6)
    assert fused[0] in {"a", "c"}
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_rrf.py -q`  
Expected: FAIL with "ImportError: cannot import name 'rrf_fuse'"

**Step 3: Write minimal implementation**
```python
# bid_scoring/search.py
def rrf_fuse(bm25_results, vector_results, k=60, bm25_weight=0.4, vector_weight=0.6):
    scores = {}
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_rrf.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add bid_scoring/search.py tests/test_rrf.py
git commit -m "feat(search): add rrf fuse"
```

---

### Task 6: MCP Server 工具实现

**自治检查（Task 6）**
- 输入：MCP SDK 已安装；数据库可连接；`chunks` 有数据与 embedding。  
- 输出：`mcp_servers/bid_documents/server.py` 与 tools JSON；`tests/test_mcp_tools.py` 通过。  
- 依赖：Task 1-5。  
- 失败与回退：失败时确认 SDK 名称与导入路径；回退可删新增文件。  

**Files:**
- Create: `mcp_servers/__init__.py`
- Create: `mcp_servers/bid_documents/__init__.py`
- Create: `mcp_servers/bid_documents/server.py`
- Create: `mcp_servers/bid_documents/tools/search_chunks.json`
- Create: `mcp_servers/bid_documents/tools/get_document_info.json`
- Create: `mcp_servers/bid_documents/tools/get_page_content.json`

**Step 1: Write the failing test**
```python
# tests/test_mcp_tools.py
from mcp_servers.bid_documents.server import search_chunks

def test_search_chunks_signature():
    assert callable(search_chunks)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_mcp_tools.py -q`  
Expected: FAIL with "ModuleNotFoundError: No module named 'mcp_servers'"

**Step 3: Write minimal implementation**
```python
# mcp_servers/__init__.py
__all__ = ["bid_documents"]
```

```python
# mcp_servers/bid_documents/__init__.py
__all__ = ["server"]
```

```python
# mcp_servers/bid_documents/server.py
# 若 MCP SDK 名称不同，请替换为实际 SDK
from fastmcp import FastMCP
import psycopg
from pgvector.psycopg import register_vector
from bid_scoring.config import load_settings
from bid_scoring.search import rrf_fuse
from bid_scoring.embeddings import embed_texts

mcp = FastMCP("bid-documents")

@mcp.tool()
def search_chunks(query: str, document_id: str, top_k: int = 10, filters: dict | None = None):
    dsn = load_settings()["DATABASE_URL"]
    filters = filters or {}
    version_id = filters.get("version_id")
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # 可按索引类型调参（择一）
            # cur.execute("SET LOCAL hnsw.ef_search = 64")
            # cur.execute("SET LOCAL ivfflat.probes = 10")
            if version_id is None:
                cur.execute(
                    """
                    select version_id
                    from document_versions
                    where doc_id = %s
                    order by created_at desc
                    limit 1
                    """,
                    (document_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return {"query": query, "results": []}
                version_id = row[0]
            query_vec = embed_texts([query])[0]
            cur.execute(
                """
                select source_id, text_raw, page_idx, bbox
                from chunks
                where version_id = %s
                  and text_tsv @@ plainto_tsquery('simple', %s)
                limit %s
                """,
                (version_id, query, top_k * 2),
            )
            bm25_rows = cur.fetchall()
            bm25 = [(r[0], i) for i, r in enumerate(bm25_rows)]
            cur.execute(
                """
                select source_id, text_raw, page_idx, bbox
                from chunks
                where version_id = %s
                order by embedding <-> %s
                limit %s
                """,
                (version_id, query_vec, top_k * 2),
            )
            vec_rows = cur.fetchall()
            vec = [(r[0], i) for i, r in enumerate(vec_rows)]
            fused_ids = rrf_fuse(bm25, vec)
    return {"query": query, "results": fused_ids[:top_k]}

@mcp.tool()
def get_document_info(document_id: str):
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select doc_id, title, source_type from documents where doc_id = %s", (document_id,))
            row = cur.fetchone()
            if row is None:
                return {"found": False}
            return {"found": True, "doc_id": row[0], "title": row[1], "source_type": row[2]}

@mcp.tool()
def get_page_content(document_id: str, page_idx: int):
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select c.text_raw, c.bbox
                from chunks c
                join document_versions dv on c.version_id = dv.version_id
                where dv.doc_id = %s and c.page_idx = %s
                limit 200
                """,
                (document_id, page_idx),
            )
            return [{"text": r[0], "bbox": r[1]} for r in cur.fetchall()]

if __name__ == "__main__":
    mcp.run()
```

```json
// mcp_servers/bid_documents/tools/search_chunks.json
{
  "name": "search_chunks",
  "description": "搜索投标文档相关片段，返回候选 source_id 列表。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "document_id": {"type": "string"},
      "top_k": {"type": "integer", "default": 10},
      "filters": {"type": "object"}
    },
    "required": ["query", "document_id"]
  }
}
```

```json
// mcp_servers/bid_documents/tools/get_document_info.json
{
  "name": "get_document_info",
  "description": "获取文档元信息。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_id": {"type": "string"}
    },
    "required": ["document_id"]
  }
}
```

```json
// mcp_servers/bid_documents/tools/get_page_content.json
{
  "name": "get_page_content",
  "description": "获取页面内容及其 bbox。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_id": {"type": "string"},
      "page_idx": {"type": "integer"}
    },
    "required": ["document_id", "page_idx"]
  }
}
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_mcp_tools.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add mcp_servers/bid_documents/server.py mcp_servers/bid_documents/tools/*.json mcp_servers/__init__.py mcp_servers/bid_documents/__init__.py tests/test_mcp_tools.py
git commit -m "feat(mcp): add search tools"
```

---

### Task 7: Skills、引用验证与 LLM 评分

**自治检查（Task 7）**
- 输入：Task 1 已完成；`references/` 目录可写；如运行真实评分需 `OPENAI_API_KEY`。  
- 输出：技能与规则文件、`bid_scoring/llm.py`、`bid_scoring/scoring.py`、`bid_scoring/verify.py`、相关脚本与测试。  
- 依赖：Task 1（配置）；可选依赖 Task 6（检索侧集成）。  
- 失败与回退：失败时检查 JSON Schema 与引用验证逻辑；回退可删新增文件。  

**Files:**
- Create: `.claude/skills/bid-scoring/SKILL.md`
- Create: `.claude/skills/citation-retriever/SKILL.md`
- Create: `.claude/skills/evidence-verifier/SKILL.md`
- Create: `references/scoring_rules.yaml`
- Create: `references/output_schema.json`
- Create: `references/output_schema_batch.json`
- Create: `bid_scoring/llm.py`
- Create: `bid_scoring/scoring.py`
- Create: `bid_scoring/verify.py`
- Create: `scripts/verify_citations.py`
- Create: `scripts/score_dimension.py`
- Create: `tests/test_llm_models.py`
- Create: `tests/test_scoring_llm.py`
- Create: `tests/test_citation_validation.py`
- Create: `tests/test_verify_citations.py`

**Step 1: Write the failing test**
```python
# tests/test_verify_citations.py
from bid_scoring.verify import verify_citation

def test_verify_exact_match():
    res = verify_citation("培训时间：2天", "培训时间：2天，含安装培训")
    assert res["verified"] is True
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_verify_citations.py -q`  
Expected: FAIL with "ModuleNotFoundError: No module named 'bid_scoring'"

**Step 3: Write minimal implementation**
```python
# bid_scoring/verify.py
import re
import unicodedata

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()

def verify_citation(cited_text: str, original_text: str):
    cited_clean = normalize(cited_text)
    original_clean = normalize(original_text)
    if cited_clean and cited_clean in original_clean:
        return {"verified": True, "match_type": "exact_normalized"}
    return {"verified": False, "match_type": "no_match"}
```

```python
# scripts/verify_citations.py
import json
from bid_scoring.verify import verify_citation

def main():
    payload = json.loads(input())
    result = verify_citation(payload["cited_text"], payload["original_text"])
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

```markdown
# .claude/skills/bid-scoring/SKILL.md
---
name: bid-scoring
description: 投标评分主技能，输出必须带引用。
---

你是投标评分专家。所有结论必须引用原文片段。
输出必须符合 references/output_schema.json。
```

```markdown
# .claude/skills/citation-retriever/SKILL.md
---
name: citation-retriever
description: 检索投标文档证据。
---

调用 MCP 工具 search_chunks 获取证据。
```

```markdown
# .claude/skills/evidence-verifier/SKILL.md
---
name: evidence-verifier
description: 引用验证。
---

调用 scripts/verify_citations.py 验证引用。
```

```yaml
# references/scoring_rules.yaml
dimensions:
  - name: 培训方案
    max_score: 10
    keywords: ["培训", "培训时间", "培训内容", "培训师"]
    rules:
      - condition: "培训方案详细，师资明确"
        score_range: [9, 10]
      - condition: "培训方案基本完整"
        score_range: [6, 8]
      - condition: "培训方案简略"
        score_range: [0, 5]
```

```json
// references/output_schema.json
{
  "type": "object",
  "properties": {
    "dimension": {"type": "string"},
    "score": {"type": "number"},
    "max_score": {"type": "number"},
    "reasoning": {"type": "string"},
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "source_number": {"type": "integer"},
          "cited_text": {"type": "string"},
          "supports_claim": {"type": "string"}
        },
        "required": ["source_number", "cited_text", "supports_claim"]
      }
    },
    "evidence_found": {"type": "boolean"}
  },
  "required": ["dimension", "score", "max_score", "reasoning", "citations", "evidence_found"]
}
```

```json
// references/output_schema_batch.json
{
  "type": "array",
  "minItems": 1,
  "items": {
    "type": "object",
    "properties": {
      "dimension": {"type": "string"},
      "score": {"type": "number"},
      "max_score": {"type": "number"},
      "reasoning": {"type": "string"},
      "citations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "source_number": {"type": "integer"},
            "cited_text": {"type": "string"},
            "supports_claim": {"type": "string"}
          },
          "required": ["source_number", "cited_text", "supports_claim"]
        }
      },
      "evidence_found": {"type": "boolean"}
    },
    "required": ["dimension", "score", "max_score", "reasoning", "citations", "evidence_found"]
  }
}
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_verify_citations.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add .claude/skills references bid_scoring/verify.py scripts/verify_citations.py tests/test_verify_citations.py
git commit -m "feat(skills): add scoring skills and citation verifier"
```

**Step 6: Write the failing tests**
```python
# tests/test_llm_models.py
from bid_scoring.llm import select_llm_model

def test_select_llm_model_uses_task_override(monkeypatch):
    monkeypatch.setenv("OPENAI_LLM_MODEL_DEFAULT", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_LLM_MODEL_SCORING", "gpt-4o")
    assert select_llm_model("scoring") == "gpt-4o"
    assert select_llm_model("unknown") == "gpt-4o-mini"
```

```python
# tests/test_scoring_llm.py
from bid_scoring.scoring import build_scoring_request

def test_build_scoring_request_uses_json_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_LLM_MODEL_DEFAULT", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_LLM_MODEL_SCORING", "gpt-4o")
    req = build_scoring_request(
        dimension="培训方案",
        max_score=10,
        evidence=["培训时间：2天，含安装培训、操作培训"],
    )
    assert req["model"] == "gpt-4o"
    assert req["response_format"]["type"] == "json_schema"
    assert req["response_format"]["json_schema"]["strict"] is True
    assert "[1]" in req["input"]
    assert "示例输出" in req["input"]
```

```python
# tests/test_citation_validation.py
from bid_scoring.verify import validate_citations

def test_validate_citations_min_required():
    evidence_map = {1: "培训时间：2天，含安装培训、操作培训"}
    citations = []
    res = validate_citations(citations, evidence_map, min_citations=1, evidence_found=True)
    assert res["valid"] is False
    assert res["reason"] == "min_citations"

def test_validate_citations_requires_empty_when_no_evidence():
    evidence_map = {1: "培训时间：2天，含安装培训、操作培训"}
    citations = [
        {
            "source_number": 1,
            "cited_text": "培训时间：2天",
            "supports_claim": "包含培训时长",
        }
    ]
    res = validate_citations(citations, evidence_map, min_citations=0, evidence_found=False)
    assert res["valid"] is False
    assert res["reason"] == "evidence_found_false_has_citations"

def test_validate_citations_allows_empty_when_no_evidence():
    evidence_map = {1: "培训时间：2天，含安装培训、操作培训"}
    res = validate_citations([], evidence_map, min_citations=0, evidence_found=False)
    assert res["valid"] is True

def test_validate_citations_happy_path():
    evidence_map = {1: "培训时间：2天，含安装培训、操作培训"}
    citations = [
        {
            "source_number": 1,
            "cited_text": "培训时间：2天",
            "supports_claim": "包含培训时长",
        }
    ]
    res = validate_citations(citations, evidence_map, min_citations=1, evidence_found=True)
    assert res["valid"] is True
```

**Step 7: Run tests to verify they fail**
Run: `pytest tests/test_llm_models.py tests/test_scoring_llm.py tests/test_citation_validation.py -q`  
Expected: FAIL with "ImportError: cannot import name 'select_llm_model'"

**Step 8: Write minimal implementation**
```python
# bid_scoring/llm.py
from openai import OpenAI
from bid_scoring.config import load_settings

def get_llm_client() -> OpenAI:
    settings = load_settings()
    timeout = settings["OPENAI_TIMEOUT"] or None
    max_retries = settings["OPENAI_MAX_RETRIES"] or None
    return OpenAI(
        api_key=settings["OPENAI_API_KEY"],
        base_url=settings["OPENAI_BASE_URL"],
        timeout=timeout,
        max_retries=max_retries,
    )

def select_llm_model(task: str) -> str:
    settings = load_settings()
    models = settings["OPENAI_LLM_MODELS"]
    return models.get(task.lower(), settings["OPENAI_LLM_MODEL_DEFAULT"])
```

```python
# bid_scoring/scoring.py
import json
from pathlib import Path
from bid_scoring.llm import get_llm_client, select_llm_model

def build_scoring_request(
    dimension: str,
    max_score: int,
    evidence: list[str],
    rules: list[str] | None = None,
) -> dict:
    schema = json.loads(Path("references/output_schema.json").read_text(encoding="utf-8"))
    evidence_lines = [f"[{i + 1}] {text}" for i, text in enumerate(evidence)]
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "（无证据）"
    rules_block = "\n- ".join(rules) if rules else "（无明确规则，需谨慎评分）"
    prompt = (
        "你是投标评分专家，只能依据给定证据评分。\n"
        "请严格输出符合 JSON Schema 的结果，只输出 JSON。\n"
        "\n"
        f"评分维度：{dimension}\n"
        f"满分：{max_score}\n"
        "评分规则：\n"
        f"- {rules_block}\n"
        "\n"
        "证据（按编号引用）：\n"
        f"{evidence_block}\n"
        "\n"
        "输出要求：\n"
        "- 输出字段必须包含：dimension, score, max_score, reasoning, citations, evidence_found\n"
        "- score 为 0 到 max_score 的数值\n"
        "- evidence_found 无证据则为 false；有有效证据才可为 true\n"
        "- citations 为数组；每条引用必须包含 source_number（证据编号）、cited_text（原文片段）、supports_claim\n"
        "- cited_text 必须是对应证据的子串\n"
        "- 无证据时 citations 为空数组\n"
        "\n"
        "示例输出（仅示例，不代表最终评分）：\n"
        "{\n"
        "  \"dimension\": \"培训方案\",\n"
        "  \"score\": 8,\n"
        "  \"max_score\": 10,\n"
        "  \"reasoning\": \"证据[1]表明包含培训时长与内容，因此方案较完整。\",\n"
        "  \"citations\": [\n"
        "    {\n"
        "      \"source_number\": 1,\n"
        "      \"cited_text\": \"培训时间：2天，含安装培训、操作培训\",\n"
        "      \"supports_claim\": \"包含培训时长与内容\"\n"
        "    }\n"
        "  ],\n"
        "  \"evidence_found\": true\n"
        "}\n"
    )
    return {
        "model": select_llm_model("scoring"),
        "input": prompt,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "bid_score",
                "schema": schema,
                "strict": True,
            },
        },
    }

def score_dimension(
    dimension: str,
    max_score: int,
    evidence: list[str],
    rules: list[str] | None = None,
    client=None,
):
    client = client or get_llm_client()
    req = build_scoring_request(dimension, max_score, evidence, rules)
    return client.responses.create(**req)
```

```python
# bid_scoring/verify.py
def validate_citations(
    citations: list[dict],
    evidence_map: dict[int, str],
    min_citations: int,
    evidence_found: bool,
) -> dict:
    if not evidence_found and citations:
        return {"valid": False, "reason": "evidence_found_false_has_citations"}
    if evidence_found and len(citations) < min_citations:
        return {"valid": False, "reason": "min_citations"}
    for item in citations:
        if not item.get("cited_text"):
            return {"valid": False, "reason": "cited_text_empty"}
        if not item.get("supports_claim"):
            return {"valid": False, "reason": "supports_claim_empty"}
        try:
            source_number = int(item.get("source_number", -1))
        except (TypeError, ValueError):
            return {"valid": False, "reason": "invalid_source_number"}
        source = evidence_map.get(source_number)
        if not source:
            return {"valid": False, "reason": "unknown_source"}
        result = verify_citation(item.get("cited_text", ""), source)
        if not result["verified"]:
            return {"valid": False, "reason": "cited_text_not_in_source"}
    return {"valid": True}
```

```python
# scripts/score_dimension.py
import json
from bid_scoring.scoring import score_dimension
from bid_scoring.verify import validate_citations

def main():
    payload = json.loads(input())
    dimension = payload["dimension"]
    max_score = payload["max_score"]
    evidence = payload.get("evidence", [])
    rules = payload.get("rules")
    min_citations = payload.get("min_citations", 1)

    response = score_dimension(dimension, max_score, evidence, rules=rules)
    result = json.loads(response.output_text)

    evidence_map = {i + 1: text for i, text in enumerate(evidence)}
    validation = validate_citations(
        result.get("citations", []),
        evidence_map,
        min_citations=min_citations,
        evidence_found=result.get("evidence_found", False),
    )
    print(json.dumps({"result": result, "validation": validation}, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

**Step 9: Run tests to verify they pass**
Run: `pytest tests/test_llm_models.py tests/test_scoring_llm.py tests/test_citation_validation.py -q`  
Expected: PASS

**Step 10: Commit**
```bash
git add bid_scoring/llm.py bid_scoring/scoring.py bid_scoring/verify.py scripts/score_dimension.py tests/test_llm_models.py tests/test_scoring_llm.py tests/test_citation_validation.py
git commit -m "feat(scoring): add llm scoring"
```

---

### Task 8: 评测回归与数据集

**自治检查（Task 8）**
- 输入：`OPENAI_API_KEY` 可用（否则评测跳过）；依赖已安装 `deepeval`。  
- 输出：`eval/dataset.jsonl`、`tests/test_eval_metrics.py`；评测通过或明确跳过。  
- 依赖：Task 1（依赖安装）。  
- 失败与回退：失败时确认网络与 API 配置；回退可删新增文件。  

**Files:**
- Create: `eval/dataset.jsonl`
- Create: `tests/test_eval_metrics.py`

**Step 1: Write the failing test**
```python
# tests/test_eval_metrics.py
import os
import pytest
if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set", allow_module_level=True)
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

def test_eval_smoke():
    case = LLMTestCase(
        input="培训方案是否完整？",
        actual_output="培训包含2天安装与操作培训。",
        expected_output="包含培训时长与内容。",
        retrieval_context=["培训时间：2天，含安装培训、操作培训"]
    )
    assert_test(case, [AnswerRelevancyMetric(threshold=0.5)])
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_eval_metrics.py -q`  
Expected: FAIL with "ModuleNotFoundError: No module named 'deepeval'"（若设置了 OPENAI_API_KEY），或 SKIP（未设置 OPENAI_API_KEY）

**Step 3: Write minimal implementation**
```json
{"input":"培训方案是否完整？","expected_output":"包含培训时长与内容。","context":["培训时间：2天，含安装培训、操作培训"]}
```

**Step 4: Run test to verify it passes**
Run: `pip install -r requirements.txt && pytest tests/test_eval_metrics.py -q`  
Expected: PASS

**Step 5: Commit**
```bash
git add eval/dataset.jsonl tests/test_eval_metrics.py
git commit -m "test(eval): add regression smoke test"
```

---

### LLM 使用约定
所有 LLM 调用必须通过配置的模型路由，避免硬编码模型名。

```python
from bid_scoring.llm import get_llm_client
from bid_scoring.scoring import build_scoring_request

client = get_llm_client()
req = build_scoring_request(
    dimension="培训方案",
    max_score=10,
    evidence=["培训时间：2天，含安装培训、操作培训"],
    rules=["培训方案详细，师资明确"],
)
response = client.responses.create(**req)
print(response.output_text)
```

---

### 多维度批量评分策略（推荐与可选）
推荐策略：对每个维度单独调用 `build_scoring_request()`，保证严格 JSON Schema 与引用校验稳定性。  
可选策略：批量评分时改用“数组 Schema”，一次请求返回多个维度结果，但需接受更高的解析与纠错成本。

**推荐（逐维调用）**
```text
对每个维度:
  1) 构建证据列表（编号从 1 开始）
  2) 调用 build_scoring_request(dimension, max_score, evidence, rules)
  3) 验证 citations（source_number 与 cited_text）
```

**可选（批量调用，需自定义 Schema）**
```text
输入模板示意：
维度 A:
  满分: 10
  规则: ...
  证据:
  [1] ...
  [2] ...
维度 B:
  满分: 5
  规则: ...
  证据:
  [1] ...

输出要求：返回数组，每个元素满足 references/output_schema.json，整体符合 references/output_schema_batch.json。
```

---

### .gitignore 建议
避免提交本地敏感配置文件。

```gitignore
.env
.env.*
!.env.example
```

## 验证清单（执行前准备）
- 本地 Postgres 已启动，且已创建 `bid_scoring` 数据库（或更新 `DATABASE_URL` 指向可用库）。  
- 设置 `DATABASE_URL` 指向本地 Postgres。  
- 准备 `.env`（可参考 `.env.example`），配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_LLM_MODEL_DEFAULT`、`OPENAI_LLM_MODEL_<TASK>`、`OPENAI_EMBEDDING_MODEL`、`OPENAI_EMBEDDING_DIM`。  
- 若运行评测，确保 `OPENAI_API_KEY` 可用，避免 deepeval 在离线环境失败。  
- 安装 pgvector 扩展（`create extension vector;`）。  
- 确保 `data/` 下存在示例 `*_content_list.json`。  
- 安装依赖：`pip install -r requirements.txt`。  

## 最终验收标准
- `pytest -q` 全绿。  
- MCP `search_chunks` 返回带 `source_id` 的结果。  
- 引用验证能稳定区分匹配与不匹配。  
- 回归评测可跑通并输出分数。  
