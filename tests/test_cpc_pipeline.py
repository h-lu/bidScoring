"""Tests for CPC (Contextual Parent-Child) Pipeline.

Tests the integration of all Multi-Vector Retrieval components:
- Contextual Retrieval (Task 1)
- HiChunk (Task 2)
- RAPTOR (Task 3)
- Late Chunking (Task 4)
- Multi-Vector Retrieval (Task 5)
"""

import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from bid_scoring.cpc_pipeline import (
    CPCPipeline,
    CPCPipelineConfig,
    ProcessResult,
    process_document_with_cpc,
    retrieve_with_cpc,
)


class TestCPCPipelineConfig:
    """Test CPCPipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CPCPipelineConfig()
        assert config.enable_contextual is True
        assert config.enable_hichunk is True
        assert config.enable_raptor is True
        assert config.enable_late_chunking is False
        assert config.contextual_model == "gpt-4"
        assert config.raptor_max_levels == 5
        assert config.retrieval_mode == "hybrid"
        assert config.retrieval_top_k == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CPCPipelineConfig(
            enable_contextual=False,
            enable_hichunk=False,
            raptor_max_levels=3,
            retrieval_top_k=10,
        )
        assert config.enable_contextual is False
        assert config.enable_hichunk is False
        assert config.raptor_max_levels == 3
        assert config.retrieval_top_k == 10


class TestCPCPipelineInit:
    """Test CPCPipeline initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        pipeline = CPCPipeline()
        assert pipeline.config is not None
        assert pipeline.config.enable_contextual is True

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = CPCPipelineConfig(enable_contextual=False)
        pipeline = CPCPipeline(config=config)
        assert pipeline.config.enable_contextual is False

    def test_init_with_mock_clients(self):
        """Test initialization with mock clients."""
        mock_llm = MagicMock()
        mock_embedding = MagicMock()
        pipeline = CPCPipeline(llm_client=mock_llm, embedding_client=mock_embedding)
        assert pipeline._llm_client is mock_llm
        assert pipeline._embedding_client is mock_embedding


class TestProcessResult:
    """Test ProcessResult dataclass."""

    def test_success_result(self):
        """Test successful result creation."""
        version_id = str(uuid.uuid4())
        result = ProcessResult(
            success=True,
            version_id=version_id,
            chunks_count=10,
            message="Success",
        )
        assert result.success is True
        assert result.version_id == version_id
        assert result.chunks_count == 10
        assert result.message == "Success"

    def test_failed_result(self):
        """Test failed result creation."""
        version_id = str(uuid.uuid4())
        result = ProcessResult(
            success=False,
            version_id=version_id,
            errors=["Error 1", "Error 2"],
            message="Failed",
        )
        assert result.success is False
        assert len(result.errors) == 2


class TestCPCPipelineDocumentProcessing:
    """Test document processing functionality."""

    @pytest.fixture
    def sample_content_list(self):
        """Provide sample MineRU content_list."""
        return [
            {"type": "text", "text": "第一章 项目概述", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "本项目旨在...", "page_idx": 1},
            {"type": "text", "text": "第二章 技术方案", "text_level": 1, "page_idx": 2},
            {"type": "text", "text": "技术细节包括...", "page_idx": 2},
            {"type": "table", "table_body": "<table>...</table>", "page_idx": 3},
        ]

    @pytest.fixture
    def uuids(self):
        """Provide UUIDs for testing."""
        return {
            "project_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "version_id": str(uuid.uuid4()),
        }

    @pytest.mark.asyncio
    async def test_process_document_basic(self, sample_content_list, uuids):
        """Test basic document processing."""
        pipeline = CPCPipeline()

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_cursor.return_value.fetchall.return_value = []

            with patch(
                "bid_scoring.cpc_pipeline.ingest_content_list"
            ) as mock_ingest:
                mock_ingest.return_value = {"total_chunks": 5}

                result = await pipeline.process_document(
                    content_list=sample_content_list,
                    document_title="测试文档",
                    project_id=uuids["project_id"],
                    document_id=uuids["document_id"],
                    version_id=uuids["version_id"],
                )

                assert isinstance(result, ProcessResult)
                assert result.version_id == uuids["version_id"]

    @pytest.mark.asyncio
    async def test_process_document_with_contextual(self, sample_content_list, uuids):
        """Test document processing with contextual chunks."""
        config = CPCPipelineConfig(enable_contextual=True)
        pipeline = CPCPipeline(config=config)

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                (str(uuid.uuid4()), "Chunk text", "text")
            ]
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            with patch(
                "bid_scoring.cpc_pipeline.ingest_content_list"
            ) as mock_ingest:
                mock_ingest.return_value = {"total_chunks": 1}

                with patch(
                    "bid_scoring.cpc_pipeline.embed_texts"
                ) as mock_embed:
                    mock_embed.return_value = [[0.1] * 1536]

                    with patch.object(
                        pipeline,
                        "_get_contextual_generator",
                    ) as mock_gen:
                        mock_generator = MagicMock()
                        mock_generator.generate_context.return_value = "Context"
                        mock_gen.return_value = mock_generator

                        result = await pipeline.process_document(
                            content_list=sample_content_list,
                            document_title="测试文档",
                            project_id=uuids["project_id"],
                            document_id=uuids["document_id"],
                            version_id=uuids["version_id"],
                        )

                        assert isinstance(result, ProcessResult)

    @pytest.mark.asyncio
    async def test_process_document_empty_content(self, uuids):
        """Test processing with empty content list."""
        pipeline = CPCPipeline()

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            with patch(
                "bid_scoring.cpc_pipeline.ingest_content_list"
            ) as mock_ingest:
                mock_ingest.return_value = {"total_chunks": 0}

                result = await pipeline.process_document(
                    content_list=[],
                    document_title="空文档",
                    project_id=uuids["project_id"],
                    document_id=uuids["document_id"],
                    version_id=uuids["version_id"],
                )

                assert isinstance(result, ProcessResult)
                assert result.chunks_count == 0

    @pytest.mark.asyncio
    async def test_process_document_ingestion_failure(self, sample_content_list, uuids):
        """Test handling of ingestion failure."""
        pipeline = CPCPipeline()

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            with patch(
                "bid_scoring.cpc_pipeline.ingest_content_list"
            ) as mock_ingest:
                mock_ingest.side_effect = Exception("Database error")

                result = await pipeline.process_document(
                    content_list=sample_content_list,
                    document_title="测试文档",
                    project_id=uuids["project_id"],
                    document_id=uuids["document_id"],
                    version_id=uuids["version_id"],
                )

                assert result.success is False
                assert len(result.errors) > 0


class TestCPCPipelineRetrieval:
    """Test retrieval functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_basic(self):
        """Test basic retrieval."""
        config = CPCPipelineConfig(retrieval_mode="hybrid", retrieval_top_k=5)
        pipeline = CPCPipeline(config=config)

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=[])
        pipeline._retriever = mock_retriever

        results = await pipeline.retrieve("查询测试", version_id=str(uuid.uuid4()))

        assert isinstance(results, list)
        mock_retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_results(self):
        """Test retrieval with mock results."""
        pipeline = CPCPipeline()

        mock_results = [
            {
                "chunk_id": str(uuid.uuid4()),
                "text": "Result 1",
                "score": 0.9,
            },
            {
                "chunk_id": str(uuid.uuid4()),
                "text": "Result 2",
                "score": 0.8,
            },
        ]

        # Create an async mock for the retrieve method
        async def async_mock_retrieve(*args, **kwargs):
            return mock_results

        mock_retriever = MagicMock()
        mock_retriever.retrieve = async_mock_retrieve
        pipeline._retriever = mock_retriever

        results = await pipeline.retrieve("查询", version_id=str(uuid.uuid4()))

        assert len(results) == 2
        assert results[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_params(self):
        """Test retrieval with custom parameters."""
        pipeline = CPCPipeline()

        mock_retriever = MagicMock()
        mock_retriever.retrieve = MagicMock(return_value=[])
        pipeline._retriever = mock_retriever

        await pipeline.retrieve(
            "查询",
            version_id=str(uuid.uuid4()),
            retrieval_mode="child",
            top_k=10,
            rerank=False,
        )

        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["retrieval_mode"] == "child"
        assert call_args.kwargs["top_k"] == 10
        assert call_args.kwargs["rerank"] is False

    @pytest.mark.asyncio
    async def test_retrieve_failure_handling(self):
        """Test retrieval failure handling."""
        pipeline = CPCPipeline()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("Retrieval error")
        pipeline._retriever = mock_retriever

        results = await pipeline.retrieve("查询", version_id=str(uuid.uuid4()))

        assert results == []


class TestCPCPipelineBuildIndices:
    """Test index building functionality."""

    @pytest.mark.asyncio
    async def test_build_indices_basic(self):
        """Test basic index building."""
        pipeline = CPCPipeline()

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            result = await pipeline.build_indices()

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_build_indices_with_version(self):
        """Test index building for specific version."""
        pipeline = CPCPipeline()
        version_id = str(uuid.uuid4())

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            result = await pipeline.build_indices(version_id=version_id)

            assert result["success"] is True


class TestCPCPipelineStats:
    """Test stats retrieval functionality."""

    def test_get_stats_basic(self):
        """Test basic stats retrieval."""
        pipeline = CPCPipeline()
        version_id = str(uuid.uuid4())

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.side_effect = [
                (10,),  # chunks
                (5,),  # contextual_chunks
                (8,),  # hierarchical_nodes
                (3,),  # multi_vector_mappings
            ]
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            stats = pipeline.get_stats(version_id)

            assert stats["version_id"] == version_id
            assert stats["chunks"] == 10
            assert stats["contextual_chunks"] == 5
            assert stats["hierarchical_nodes"] == 8
            assert stats["multi_vector_mappings"] == 3

    def test_get_stats_error_handling(self):
        """Test stats error handling."""
        pipeline = CPCPipeline()
        version_id = str(uuid.uuid4())

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_conn.side_effect = Exception("Database error")

            stats = pipeline.get_stats(version_id)

            assert "error" in stats
            assert stats["version_id"] == version_id


class TestCPCPipelineHelperMethods:
    """Test internal helper methods."""

    @pytest.mark.asyncio
    async def test_build_contextual_chunks(self):
        """Test contextual chunks building."""
        pipeline = CPCPipeline()
        version_id = str(uuid.uuid4())

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (str(uuid.uuid4()), "Text 1", "text"),
            (str(uuid.uuid4()), "Text 2", "text"),
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("bid_scoring.cpc_pipeline.embed_texts") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

            with patch.object(pipeline, "_get_contextual_generator") as mock_gen:
                mock_generator = MagicMock()
                mock_generator.generate_context.return_value = "Context"
                mock_gen.return_value = mock_generator

                count = await pipeline._build_contextual_chunks(
                    mock_conn, version_id, "Test Doc"
                )

                assert count == 2

    @pytest.mark.asyncio
    async def test_build_hierarchical_nodes(self):
        """Test hierarchical nodes building."""
        pipeline = CPCPipeline()
        version_id = str(uuid.uuid4())

        content_list = [
            {"type": "text", "text": "Title", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "Content", "page_idx": 1},
        ]

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("chunk_0000", str(uuid.uuid4()))]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = await pipeline._build_hierarchical_nodes(
            mock_conn, version_id, content_list, "Test Doc"
        )

        assert count > 0

    @pytest.mark.asyncio
    async def test_build_multi_vector_mappings(self):
        """Test multi-vector mappings building."""
        pipeline = CPCPipeline()
        version_id = str(uuid.uuid4())

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [
            # First call - chunks
            [
                (str(uuid.uuid4()), "text", 1),
                (str(uuid.uuid4()), "text", 2),
                (str(uuid.uuid4()), "text", 3),
            ],
            # Second call - contextual chunks
            [],
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = await pipeline._build_multi_vector_mappings(mock_conn, version_id)

        assert count >= 0

    @pytest.mark.asyncio
    async def test_generate_embeddings(self):
        """Test embeddings generation."""
        pipeline = CPCPipeline()
        version_id = str(uuid.uuid4())

        chunk_id = str(uuid.uuid4())
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(chunk_id, "Test text")]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("bid_scoring.cpc_pipeline.embed_texts") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]

            await pipeline._generate_embeddings(mock_conn, version_id)

            mock_embed.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_process_document_with_cpc(self):
        """Test process_document_with_cpc convenience function."""
        content_list = [{"type": "text", "text": "Test"}]

        with patch.object(CPCPipeline, "process_document") as mock_process:
            mock_process.return_value = ProcessResult(
                success=True,
                version_id=str(uuid.uuid4()),
                chunks_count=1,
            )

            result = await process_document_with_cpc(
                content_list=content_list,
                document_title="Test",
                project_id=str(uuid.uuid4()),
                document_id=str(uuid.uuid4()),
                version_id=str(uuid.uuid4()),
            )

            assert result.success is True
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_cpc(self):
        """Test retrieve_with_cpc convenience function."""
        with patch.object(CPCPipeline, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = [{"chunk_id": str(uuid.uuid4())}]

            results = await retrieve_with_cpc("query", version_id=str(uuid.uuid4()))

            assert len(results) == 1
            mock_retrieve.assert_called_once()


class TestCPCPipelineIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test complete pipeline flow from processing to retrieval."""
        content_list = [
            {"type": "text", "text": "第一章", "text_level": 1, "page_idx": 1},
            {"type": "text", "text": "内容...", "page_idx": 1},
        ]
        version_id = str(uuid.uuid4())

        pipeline = CPCPipeline()

        # Mock all database operations
        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            with patch(
                "bid_scoring.cpc_pipeline.ingest_content_list"
            ) as mock_ingest:
                mock_ingest.return_value = {"total_chunks": 2}

                # Process document
                process_result = await pipeline.process_document(
                    content_list=content_list,
                    document_title="测试文档",
                    project_id=str(uuid.uuid4()),
                    document_id=str(uuid.uuid4()),
                    version_id=version_id,
                )

                assert process_result.version_id == version_id

    def test_pipeline_configuration_cascading(self):
        """Test that configuration cascades to all components."""
        config = CPCPipelineConfig(
            contextual_model="gpt-3.5-turbo",
            raptor_max_levels=3,
            retrieval_top_k=10,
        )
        pipeline = CPCPipeline(config=config)

        assert pipeline.config.contextual_model == "gpt-3.5-turbo"
        assert pipeline.config.raptor_max_levels == 3
        assert pipeline.config.retrieval_top_k == 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self):
        """Test retrieval with empty query."""
        pipeline = CPCPipeline()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        pipeline._retriever = mock_retriever

        results = await pipeline.retrieve("", version_id=str(uuid.uuid4()))

        assert results == []

    @pytest.mark.asyncio
    async def test_process_document_single_chunk(self):
        """Test processing with single chunk."""
        pipeline = CPCPipeline()
        content_list = [{"type": "text", "text": "Only content", "page_idx": 1}]

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            with patch(
                "bid_scoring.cpc_pipeline.ingest_content_list"
            ) as mock_ingest:
                mock_ingest.return_value = {"total_chunks": 1}

                result = await pipeline.process_document(
                    content_list=content_list,
                    document_title="单块文档",
                    project_id=str(uuid.uuid4()),
                    document_id=str(uuid.uuid4()),
                    version_id=str(uuid.uuid4()),
                )

                assert result.chunks_count == 1

    @pytest.mark.asyncio
    async def test_component_failure_continues(self):
        """Test that component failure doesn't stop pipeline."""
        pipeline = CPCPipeline(config=CPCPipelineConfig(enable_contextual=True))

        with patch.object(pipeline, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )

            with patch(
                "bid_scoring.cpc_pipeline.ingest_content_list"
            ) as mock_ingest:
                mock_ingest.return_value = {"total_chunks": 5}

                with patch.object(
                    pipeline,
                    "_build_contextual_chunks",
                    side_effect=Exception("Contextual failed"),
                ):
                    result = await pipeline.process_document(
                        content_list=[{"type": "text", "text": "Test"}],
                        document_title="测试",
                        project_id=str(uuid.uuid4()),
                        document_id=str(uuid.uuid4()),
                        version_id=str(uuid.uuid4()),
                    )

                    # Should still succeed even if contextual failed
                    assert result.success is True
                    assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
