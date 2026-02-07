# ColBERT Reranker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ColBERT as an optional reranker backend in hybrid retrieval while preserving current cross-encoder default behavior.

**Architecture:** Keep `HybridRetriever` as the orchestrator and introduce a backend-agnostic reranker interface. Implement a new ColBERT reranker adapter using RAGatouille with graceful fallback when dependency is absent. Wire backend selection through retriever init flags and keep sync/async retrieval paths consistent.

**Tech Stack:** Python, `bid_scoring/hybrid_retrieval.py`, optional `ragatouille` (ColBERT), existing `sentence-transformers` CrossEncoder, pytest, ruff.

### Task 1: Add failing tests for backend selection and ColBERT rerank wiring

**Files:**
- Modify: `tests/test_hybrid_retrieval.py`

**Step 1: Write failing test for backend selection with `rerank_backend="colbert"`**

```python
def test_colbert_backend_selected_when_enabled():
    ...
```

**Step 2: Write failing test for sync retrieve path calling ColBERT reranker**

```python
def test_retrieve_applies_colbert_reranker_when_selected():
    ...
```

**Step 3: Write failing test for async retrieve path calling ColBERT reranker**

```python
@pytest.mark.asyncio
async def test_retrieve_async_applies_colbert_reranker_when_selected():
    ...
```

**Step 4: Run focused tests to confirm RED**

Run: `pytest -q tests/test_hybrid_retrieval.py -k "colbert_backend_selected or colbert_reranker"`
Expected: FAIL because ColBERT backend selection is not implemented yet.

### Task 2: Implement ColBERT reranker adapter and backend routing

**Files:**
- Modify: `bid_scoring/hybrid_retrieval.py`

**Step 1: Add optional import guard for RAGatouille**

```python
try:
    from ragatouille import RAGPretrainedModel
    HAS_COLBERT_RERANKER = True
except ImportError:
    HAS_COLBERT_RERANKER = False
```

**Step 2: Implement `ColBERTReranker` class with `rerank(query, results, top_n)`**

```python
class ColBERTReranker:
    DEFAULT_MODEL = "colbert-ir/colbertv2.0"
    ...
```

**Step 3: Add retriever init config for backend selection**

```python
rerank_backend: str = "cross_encoder"
```

**Step 4: Route reranker initialization based on backend and availability**

- `cross_encoder`: existing behavior
- `colbert`: initialize `ColBERTReranker` when installed
- unavailable backend dependency: log warning and disable rerank

**Step 5: Keep retrieve/retrieve_async rerank invocation backend-agnostic**

- no API change in call sites beyond selected `_reranker` instance

### Task 3: Validate behavior and compatibility

**Files:**
- Modify: `tests/test_hybrid_retrieval.py`

**Step 1: Run focused ColBERT tests**

Run: `pytest -q tests/test_hybrid_retrieval.py -k "colbert_backend_selected or colbert_reranker"`
Expected: PASS.

**Step 2: Run full hybrid retrieval test file**

Run: `pytest -q tests/test_hybrid_retrieval.py`
Expected: PASS.

**Step 3: Run lint and format checks**

Run: `ruff check bid_scoring/ tests/ && ruff format --check bid_scoring/ tests/`
Expected: PASS.

### Task 4: Document operational guidance inline

**Files:**
- Modify: `bid_scoring/hybrid_retrieval.py`

**Step 1: Update docstring for rerank parameters to include backend choices**
**Step 2: Add warning message with install hint for missing `ragatouille`**

### Task 5: Commit as atomic change

**Files:**
- Modify: `bid_scoring/hybrid_retrieval.py`
- Modify: `tests/test_hybrid_retrieval.py`
- Create: `docs/plans/2026-02-06-colbert-reranker.md`

**Step 1: Stage files**

Run: `git add bid_scoring/hybrid_retrieval.py tests/test_hybrid_retrieval.py docs/plans/2026-02-06-colbert-reranker.md`

**Step 2: Commit**

Run: `git commit -m "feat(retrieval): add colbert rerank backend"`
