# Bid Scoring Retrieval MCP Server

> MCP Server for automated bid/tender document analysis.  
> Designed for Claude Code and AI agents.

## Quick Start

### Option 1: Run from local source

```bash
# Start the server
uv run fastmcp run mcp_servers/retrieval_server.py -t stdio

# Or with HTTP transport
uv run fastmcp run mcp_servers/retrieval_server.py -t http --port 8000
```

### Option 2: Install via uvx from GitHub (Recommended for MCP clients)

```bash
# Run directly without cloning
uvx --from git+https://github.com/h-lu/bidScoring.git bid-scoring-retrieval

# Or specify a specific version/branch
uvx --from git+https://github.com/h-lu/bidScoring.git@main bid-scoring-retrieval
```

### Claude Code MCP Configuration

Add to your Claude Code MCP config (`~/.claude/mcp.json` or project-specific):

```json
{
  "mcpServers": {
    "bid-scoring": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/h-lu/bidScoring.git",
        "bid-scoring-retrieval"
      ],
      "env": {
        "DATABASE_URL": "postgresql://localhost:5432/bid_scoring",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Environment Variables

```bash
DATABASE_URL="postgresql://localhost:5432/bid_scoring"
OPENAI_API_KEY="sk-..."  # Required for vector search
BID_SCORING_RETRIEVER_CACHE_SIZE="32"
BID_SCORING_QUERY_CACHE_SIZE="1024"
```

## Tools Overview

This MCP server provides **12 tools** organized in 6 layers for bid analysis workflows:

### Layer 1: Discovery (发现)
Explore available documents and understand structure.

| Tool | Purpose | Key Use Case |
|------|---------|--------------|
| `list_available_versions` | List all bidding documents | Start here to see what's available |
| `get_document_outline` | Get document TOC/structure | Find where "售后服务" section is |
| `get_page_metadata` | Get page composition stats | Check if page 17 has tables |

### Layer 2: Search (检索)
Find relevant content with precision.

| Tool | Purpose | Key Use Case |
|------|---------|--------------|
| `retrieve` | Basic retrieval (backward compatible) | Simple search |
| `search_chunks` | Advanced search with filters | Search only pages 15-25 |
| `search_by_heading` | Find by section heading | Get entire "售后服务方案" section |

### Layer 3: Filter (过滤)
Refine search results.

| Tool | Purpose | Key Use Case |
|------|---------|--------------|
| `filter_and_sort_results` | Filter/sort results | Keep only results with score > 0.6 |

### Layer 4: Batch (批量)
Analyze multiple dimensions at once.

| Tool | Purpose | Key Use Case |
|------|---------|--------------|
| `batch_search` | Multi-query batch search | Search "质保期", "响应时间", "培训天数" together |

### Layer 5: Evidence (证据)
Get precise, verifiable evidence.

| Tool | Purpose | Key Use Case |
|------|---------|--------------|
| `get_chunk_with_context` | Get chunk with surrounding context | Avoid misinterpreting table cells |
| `get_unit_evidence` | Get audit-grade evidence | Verify exact quote with hash |

### Layer 6: Compare (对比)
Compare across multiple bidders.

| Tool | Purpose | Key Use Case |
|------|---------|--------------|
| `compare_across_versions` | Cross-bidder comparison | Compare how A/B/C respond to same requirement |
| `extract_key_value` | Extract structured commitments | Extract "质保期: 5年" format |

## Common Workflows

### Workflow 1: Analyze a Single Bidder

```python
# Step 1: List available documents
versions = list_available_versions(include_stats=True)
# Returns: version_id, title, chunk_count, page_count

# Step 2: Get document structure
outline = get_document_outline(version_id="uuid", max_depth=2)
# Returns: sections with page ranges

# Step 3: Search specific section
results = search_chunks(
    version_id="uuid",
    query="售后响应时间",
    page_range=[15, 25],  # Focus on relevant pages
    top_k=5
)

# Step 4: Get full context for top result
detail = get_chunk_with_context(
    chunk_id=results["results"][0]["chunk_id"],
    context_depth="section"
)

# Step 5: Extract structured data
commitments = extract_key_value(
    version_id="uuid",
    key_patterns=["响应时间", "到场时间"],
    value_patterns=["小时", "分钟"]
)
```

### Workflow 2: Compare Multiple Bidders

```python
# Step 1: Compare same query across versions
comparison = compare_across_versions(
    version_ids=["uuid-a", "uuid-b", "uuid-c"],
    query="售后服务响应时间 SLA",
    normalize_scores=True
)
# Returns: Results from each version with normalized scores

# Step 2: Extract commitments from each
for version_id in ["uuid-a", "uuid-b", "uuid-c"]:
    data = extract_key_value(
        version_id=version_id,
        key_patterns=["质保期", "保修期"],
        value_patterns=["年", "月"]
    )
    # Compare extracted values
```

### Workflow 3: Batch Analysis

```python
# Analyze multiple dimensions at once
batch_results = batch_search(
    version_id="uuid",
    queries=[
        "售后响应时间",
        "质保期限",
        "工程师团队",
        "备件库存策略",
        "故障处理流程"
    ],
    top_k_per_query=3,
    aggregate_by="page"  # Group by page for reading
)
```

## Tool Reference

### list_available_versions

List all available document versions with metadata.

**Parameters:**
- `project_id` (string, optional): Filter by project
- `include_stats` (boolean, default: true): Include chunk/page counts

**Returns:**
```json
{
  "count": 3,
  "versions": [
    {
      "version_id": "33333333-3333-3333-3333-333333333333",
      "title": "上海澄研医疗科技投标文件",
      "chunk_count": 1007,
      "page_count": 45
    }
  ]
}
```

**Example:**
```python
versions = list_available_versions(include_stats=True)
for v in versions["versions"]:
    print(f"{v['title']}: {v['chunk_count']} chunks")
```

---

### get_document_outline

Get document structure (table of contents).

**Parameters:**
- `version_id` (string, required): Document UUID
- `max_depth` (integer, default: 3): Hierarchy depth

**Returns:**
```json
{
  "version_id": "uuid",
  "source": "hierarchical_nodes",
  "outline": [
    {
      "node_id": "...",
      "level": 2,
      "heading": "售后服务方案",
      "page_range": [16, 19]
    }
  ]
}
```

**Example:**
```python
outline = get_document_outline(version_id="uuid", max_depth=2)
for item in outline["outline"]:
    if "售后" in item["heading"]:
        print(f"售后服务 at pages {item['page_range']}")
```

---

### search_chunks

Advanced search with filtering.

**Parameters:**
- `version_id` (string, required): Document UUID
- `query` (string, required): Search query
- `top_k` (integer, default: 10): Max results
- `mode` (enum: "hybrid" | "vector" | "keyword", default: "hybrid")
- `page_range` (tuple[int, int], optional): Limit to page range
- `element_types` (list[string], optional): Filter by type

**Returns:**
```json
{
  "version_id": "uuid",
  "query": "售后响应时间",
  "results": [
    {
      "chunk_id": "...",
      "page_idx": 17,
      "score": 0.95,
      "text": "响应时效：2小时内..."
    }
  ]
}
```

**Example:**
```python
results = search_chunks(
    version_id="uuid",
    query="售后响应时间",
    page_range=[15, 25],  # Only search relevant pages
    top_k=5
)
```

---

### compare_across_versions

Compare responses across multiple bidders.

**Parameters:**
- `version_ids` (list[string], required): List of version UUIDs
- `query` (string, required): Search query
- `top_k_per_version` (integer, default: 3)
- `normalize_scores` (boolean, default: true)

**Returns:**
```json
{
  "query": "售后响应时间",
  "versions_compared": ["uuid-a", "uuid-b", "uuid-c"],
  "results_by_version": {
    "uuid-a": [{"text": "2小时响应...", "score": 0.95}],
    "uuid-b": [{"text": "4小时响应...", "score": 0.88}]
  }
}
```

**Example:**
```python
comparison = compare_across_versions(
    version_ids=["uuid-a", "uuid-b", "uuid-c"],
    query="售后服务响应时间",
    normalize_scores=True
)
for version_id, results in comparison["results_by_version"].items():
    print(f"{version_id}: {results[0]['text'][:50]}")
```

---

### extract_key_value

Extract structured commitments.

**Parameters:**
- `version_id` (string, required)
- `key_patterns` (list[string], required): Keywords to search
- `value_patterns` (list[string], optional): Value patterns
- `fuzzy_match` (boolean, default: true)
- `context_window` (integer, default: 50)

**Returns:**
```json
[
  {
    "key": "质保期",
    "value": "5年",
    "numeric_value": 5,
    "unit": "年",
    "page_idx": 16,
    "context": "...质保期：原厂免费质保5年..."
  }
]
```

**Example:**
```python
commitments = extract_key_value(
    version_id="uuid",
    key_patterns=["质保期", "保修期"],
    value_patterns=["年", "月", "天"]
)
for c in commitments:
    print(f"{c['key']}: {c['value']} (page {c['page_idx']})")
```

---

### get_chunk_with_context

Get chunk with surrounding context.

**Parameters:**
- `chunk_id` (string, required)
- `context_depth` (enum: "chunk" | "paragraph" | "section" | "document", default: "paragraph")
- `include_adjacent_pages` (boolean, default: false)

**Returns:**
```json
{
  "chunk_id": "...",
  "text": "...",
  "page_idx": 17,
  "hierarchy": [...],
  "same_page_chunks": [...]
}
```

**Example:**
```python
detail = get_chunk_with_context(
    chunk_id="uuid",
    context_depth="section",  # Get entire section
    include_adjacent_pages=True
)
```

---

## Version IDs for Testing

| Scenario | Version ID | Bidder Name |
|----------|-----------|-------------|
| A | `33333333-3333-3333-3333-333333333333` | 上海澄研医疗科技 |
| B | `44444444-4444-4444-4444-444444444444` | 苏州启衡生物仪器 |
| C | `55555555-5555-5555-5555-555555555555` | 杭州赛泓精密医疗 |

## Tips for Claude Code

### 1. Always Start with Discovery
```python
# Good: First check what's available
versions = list_available_versions(include_stats=True)
outline = get_document_outline(version_id=versions["versions"][0]["version_id"])

# Then search with context
results = search_chunks(...)
```

### 2. Use Page Range Filtering
```python
# Good: Narrow search to relevant pages
results = search_chunks(
    version_id="uuid",
    query="售后服务",
    page_range=[15, 25]  # Known from outline
)
```

### 3. Get Context for Important Results
```python
# Good: Always get context before quoting
top_result = results["results"][0]
detail = get_chunk_with_context(
    chunk_id=top_result["chunk_id"],
    context_depth="section"
)
# Now you have full context for accurate analysis
```

### 4. Batch Related Queries
```python
# Good: Batch related dimensions
batch_results = batch_search(
    version_id="uuid",
    queries=["响应时间", "到场时间", "维修时效"],
    aggregate_by="page"
)
```

### 5. Compare Across Bidders
```python
# Good: Always compare multiple bidders for key metrics
comparison = compare_across_versions(
    version_ids=["uuid-a", "uuid-b", "uuid-c"],
    query="质保期限",
    normalize_scores=True
)
```

## Error Handling

Tools return standardized error responses:

```json
{
  "success": false,
  "error": "version_id is required and cannot be empty",
  "execution_time_ms": 0.5
}
```

Common errors:
- `version_id is required`: Missing required parameter
- `version_id not found`: Invalid UUID
- `query is required`: Missing search query
- Database connection errors

## Performance Notes

| Operation | Typical Latency |
|-----------|----------------|
| `list_available_versions` | ~100ms |
| `get_document_outline` | ~200ms |
| `search_chunks` (hybrid) | ~1.5s |
| `search_chunks` (keyword) | ~10ms |
| `search_chunks` (vector) | ~800ms |
| `batch_search` (4 queries) | ~3s |
| `compare_across_versions` (3 versions) | ~4s |

## Development

```bash
# Run tests
uv run pytest tests/test_mcp_retrieval_server.py -v

# Inspect server
uv run fastmcp inspect mcp_servers/retrieval_server.py

# Run with HTTP for debugging
uv run fastmcp run mcp_servers/retrieval_server.py -t http --port 8000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code / AI Agent                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               MCP Server (Bid Scoring Retrieval)             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Discovery  │ │   Search    │ │   Compare   │           │
│  │  (3 tools)  │ │  (3 tools)  │ │  (2 tools)  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Filter    │ │    Batch    │ │  Evidence   │           │
│  │  (1 tool)   │ │  (1 tool)   │ │  (2 tools)  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PostgreSQL + pgvector + HiChunk                 │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT License - See LICENSE file
