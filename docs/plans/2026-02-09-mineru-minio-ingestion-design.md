# MinerU + MinIO + 数据库集成设计文档

**日期**: 2026-02-09
**作者**: Claude Code
**状态**: 设计阶段

---

## 1. 概述

### 1.1 目标

将原始 PDF 文档通过 MinerU API 处理后，实现以下功能：

1. 将处理结果（原始 PDF + 解析文件）存储到 MinIO
2. 将 `content_list.json` 导入 PostgreSQL 数据库
3. 自动执行 embedding 向量化
4. 构建 HiChunk 分层索引结构

### 1.2 整体架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  原始 PDF 文件   │ ──> │  MinerU API    │ ──> │  解析结果 ZIP   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PostgreSQL DB  │ ◀─ │  处理协调器     │ ──> │  MinIO 存储     │
│                 │     │  (Coordinator)  │     │                 │
│  - documents    │     └─────────────────┘     │  bids/{proj_id} │
│  - chunks       │              │               │    /{version_id} │
│  - embeddings   │              ▼               │    /files/...   │
│                 │     ┌─────────────────┐     └─────────────────┘
│  - hichunk_nodes│     │  向量化服务     │
│                 │     │  (Embedder)     │
└─────────────────┘     └─────────────────┘
```

---

## 2. 数据模型

### 2.1 新增 `document_files` 表

```sql
-- 文件存储元数据表
CREATE TABLE document_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    file_type TEXT NOT NULL,              -- 'original_pdf', 'parsed_zip', 'markdown', 'image', 'json'
    file_path TEXT NOT NULL,              -- MinIO object key: bids/{project_id}/{version_id}/files/...
    file_name TEXT NOT NULL,              -- 原始文件名
    file_size BIGINT,                     -- 文件大小（字节）
    content_type TEXT,                    -- MIME type
    etag TEXT,                            -- MinIO ETag (MD5)
    metadata JSONB,                       -- 额外元数据
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (version_id, file_path)
);

-- 索引
CREATE INDEX idx_document_files_version ON document_files(version_id);
CREATE INDEX idx_document_files_type ON document_files(file_type);
```

### 2.2 扩展 `chunks` 表

```sql
-- 添加向量化状态字段
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_status TEXT DEFAULT 'pending';
-- 可能值: 'pending', 'processing', 'completed', 'failed'

CREATE INDEX idx_chunks_embedding_status ON chunks(embedding_status)
WHERE embedding_status != 'completed';
```

### 2.3 MinIO 存储路径规范

```
bids/
└── {project_id}/
    └── {version_id}/
        └── files/
            ├── original/
            │   └── {original_filename}.pdf
            ├── parsed/
            │   ├── full.md
            │   ├── content_list.json
            │   ├── images/
            │   │   ├── img_001.png
            │   │   └── ...
            │   └── ...
            └── archive/
                └── {batch_id}.zip        -- 原始 ZIP 包备份
```

---

## 3. 核心组件

### 3.1 组件清单

| 组件 | 文件路径 | 状态 | 说明 |
|------|----------|------|------|
| MinerU API Client | `mineru/process_pdfs.py` | 已存在 | 调用 MinerU Cloud API |
| MinIO Storage Module | `mineru/minio_storage.py` | 新增 | MinIO 上传/下载管理 |
| Database Ingest Module | `bid_scoring/ingest.py` | 已存在 | content_list 入库 |
| File Registry Module | `bid_scoring/files.py` | 新增 | document_files 表管理 |
| Embedding Batch Service | `bid_scoring/embeddings_batch.py` | 新增 | 批量向量化 |
| HiChunk Builder | `bid_scoring/hichunk_builder.py` | 已存在 | 分层索引构建 |
| Processing Coordinator | `mineru/coordinator.py` | 新增 | 流程协调 |

### 3.2 MinIO Storage Module (`mineru/minio_storage.py`)

```python
class MinIOStorage:
    """MinIO 对象存储管理"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        ...

    def upload_file(self, local_path: Path, object_key: str, metadata: dict) -> FileRecord
        """上传单个文件，返回文件记录"""

    def upload_directory(self, local_dir: Path, prefix: str) -> list[FileRecord]
        """递归上传目录下所有文件"""

    def generate_presigned_url(self, object_key: str, expires: timedelta) -> str
        """生成预签名下载 URL"""

    def list_files(self, prefix: str) -> list[ObjectInfo]
        """列出指定前缀的所有文件"""
```

### 3.3 File Registry Module (`bid_scoring/files.py`)

```python
class FileRegistry:
    """文件元数据注册中心"""

    def register_file(self, version_id: str, file_type: str, file_path: str,
                      file_name: str, file_size: int, content_type: str,
                      etag: str, metadata: dict) -> str
        """注册文件记录，返回 file_id"""

    def get_files_by_version(self, version_id: str) -> list[FileRecord]
        """获取指定版本的所有文件"""

    def get_original_pdf(self, version_id: str) -> FileRecord | None
        """获取原始 PDF 文件记录"""
```

### 3.4 Embedding Batch Service (`bid_scoring/embeddings_batch.py`)

```python
class EmbeddingBatchService:
    """批量向量化服务"""

    BATCH_SIZE = 100  # 每批处理 100 条
    MAX_RETRIES = 6   # 最大重试次数

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(MAX_RETRIES))
    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]
        """批量获取 embeddings，带重试"""

    def process_version(self, version_id: str, conn) -> EmbeddingReport
        """处理指定版本的所有 pending chunks"""

    def process_pending(self, limit: int = 1000, conn) -> EmbeddingReport
        """处理全局 pending chunks"""
```

### 3.5 Processing Coordinator (`mineru/coordinator.py`)

```python
class ProcessingCoordinator:
    """处理流程协调器"""

    def process_pdf(self, pdf_path: Path, project_id: str | None = None) -> ProcessingResult
        """完整的 PDF 处理流程:

        1. 调用 MinerU API 解析 PDF
        2. 上传所有文件到 MinIO
        3. 导入 content_list.json 到数据库
        4. 触发向量化任务
        5. 构建 HiChunk 索引
        """

    def process_existing_output(self, output_dir: Path, project_id: str,
                                document_id: str, version_id: str) -> ProcessingResult
        """处理已存在的 MinerU 输出目录"""
```

---

## 4. 处理流程

### 4.1 完整处理流程

```python
# 伪代码
def process_pdf(pdf_path: Path) -> ProcessingResult:
    # Step 1: MinerU API 解析
    zip_path = call_mineru_api(pdf_path)
    extract_dir = extract_zip(zip_path)

    # Step 2: 生成/获取 IDs
    project_id = generate_or_get_project_id()
    document_id = generate_document_id(project_id, pdf_path.name)
    version_id = generate_version_id()

    # Step 3: 上传到 MinIO
    minio = MinIOStorage(...)
    files = minio.upload_directory(extract_dir, f"{project_id}/{version_id}/files/")

    # Step 4: 注册文件元数据
    registry = FileRegistry(conn)
    for f in files:
        registry.register_file(version_id, ...)

    # Step 5: 导入 content_list.json
    content_list = load_json(extract_dir / "content_list.json")
    ingest_content_list(conn, project_id, document_id, version_id, content_list)

    # Step 6: 批量向量化
    embedder = EmbeddingBatchService(...)
    embedder.process_version(version_id, conn)

    # Step 7: 构建 HiChunk
    hichunk_builder.build(version_id, conn)

    return ProcessingResult(
        project_id=project_id,
        document_id=document_id,
        version_id=version_id,
        status="completed"
    )
```

### 4.2 流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         process_pdf()                                │
└─────────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
            ┌─────────────┐       ┌─────────────┐
            │ MinerU API  │       │  生成 IDs    │
            │   解析 PDF   │       │  (proj/doc/  │
            └─────────────┘       │   version)   │
                    │             └─────────────┘
                    ▼                     │
            ┌─────────────┐               │
            │  下载 ZIP   │               │
            │  并解压      │               │
            └─────────────┘               │
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────┐
                    │  上传 MinIO     │
                    │  (所有文件)      │
                    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │  注册文件元数据  │
                    │ (document_files) │
                    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │ 导入 content_   │
                    │ list.json      │
                    │ (chunks/units)  │
                    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │  批量向量化     │
                    │  (embeddings)   │
                    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │  构建 HiChunk   │
                    │  (分层索引)      │
                    └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │  返回 Result    │
                    └─────────────────┘
```

---

## 5. 环境变量配置

### 5.1 新增环境变量

```bash
# .env 新增
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=bids
MINIO_SECURE=false

# Embedding 批处理配置
EMBEDDING_BATCH_SIZE=100
EMBEDDING_MAX_RETRIES=6
```

---

## 6. CLI 接口

### 6.1 新增命令

```bash
# 处理单个 PDF
python -m mineru.coordinator process /path/to/file.pdf

# 处理整个目录
python -m mineru.coordinator process-dir /path/to/pdf/dir

# 处理已有的 MinerU 输出
python -m mineru.coordinator ingest-existing /path/to/output

# 向量化任务
python -m bid_scoring.embeddings_batch process-version <version_id>
python -m bid_scoring.embeddings_batch process-pending --limit 1000
```

---

## 7. 错误处理

### 7.1 重试策略

| 操作 | 重试策略 | 说明 |
|------|----------|------|
| MinerU API 上传 | 指数退避, 3 次 | 已实现 |
| MinIO 上传 | 指数退避, 3 次 | 新增 |
| Embedding API | 指数退避, 6 次 | tenacity |
| 数据库插入 | 事务回滚 | 单条失败不影响其他 |

### 7.2 状态管理

- 每个 chunk 有 `embedding_status` 字段
- 支持断点续传
- 失败任务可重试

---

## 8. 后续扩展

### 8.1 潜在改进

1. **异步任务队列**: 使用 Celery/ARQ 实现后台处理
2. **进度通知**: WebSocket 推送处理进度
3. **文件生命周期**: 自动清理过期文件
4. **增量更新**: 检测 PDF 变更只更新变更部分

### 8.2 监控指标

- 处理成功率
- 平均处理时间
- MinIO 存储使用量
- Embedding API 调用次数
