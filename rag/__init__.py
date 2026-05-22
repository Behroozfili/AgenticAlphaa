# rag/__init__.py
from rag.loader import AlphaLoader, RawDocument
from rag.processor import AlphaProcessor, ProcessedChunk, ProcessorMetrics
from rag.embedding_manager import AlphaEmbedder, get_embedder
from rag.vector_store import AlphaVectorStore
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
    "AlphaLoader", "RawDocument",
    # Processor
    "AlphaProcessor", "ProcessedChunk", "ProcessorMetrics",
    # Embedder
    "AlphaEmbedder", "get_embedder",
    # Stores
    "AlphaVectorStore",
    "AlphaGraphStore", "GraphDocument", "Entity", "Relation",
    # Retriever
    "AlphaRetriever",
    # Evaluator
    "AlphaEvaluator", "EvaluationReport",
    # Hybrid RAG tools
    "rag_vector_search",
    "rag_graph_traverse",
    "rag_hybrid_query",
]