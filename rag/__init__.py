# rag/__init__.py
from rag.loader import AlphaLoader, HistoricalSourceUnavailable, QuarantinedDocument, RawDocument
from rag.processor import AlphaProcessor, ProcessedChunk, ProcessorMetrics
from rag.embedding_manager import AlphaEmbedder, EmbeddingManifest, get_embedder
from rag.vector_store import AlphaVectorStore, TemporalBoundaryUnsupported, VectorStoreConfig
from rag.ingestion import CheckpointStore, IngestionManifest, StageStatus, run_ingestion_pipeline
from rag.graph_store import AlphaGraphStore, GraphDocument, Entity, Relation
from rag.retriever import AlphaRetriever
from rag.evaluation import AlphaEvaluator, EvaluationReport
from rag.hybrid_rag import (
    rag_vector_search,
    rag_graph_traverse,
    rag_hybrid_query,
)

__all__ = [
    # Loader
    "AlphaLoader", "HistoricalSourceUnavailable", "QuarantinedDocument", "RawDocument",
    # Processor
    "AlphaProcessor", "ProcessedChunk", "ProcessorMetrics",
    # Embedder
    "AlphaEmbedder", "EmbeddingManifest", "get_embedder",
    # Stores
    "AlphaVectorStore", "TemporalBoundaryUnsupported", "VectorStoreConfig",
    "AlphaGraphStore", "GraphDocument", "Entity", "Relation",
    # Retriever
    "AlphaRetriever",
    # Ingestion
    "CheckpointStore", "IngestionManifest", "StageStatus", "run_ingestion_pipeline",
    # Evaluator
    "AlphaEvaluator", "EvaluationReport",
    # Hybrid RAG tools
    "rag_vector_search",
    "rag_graph_traverse",
    "rag_hybrid_query",
]
