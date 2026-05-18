# ingestion.py
import logging
import os

# Assuming the rag package is in the current directory or Python path
from rag.loader import AlphaLoader, RawDocument
from rag.processor import AlphaProcessor, ProcessedChunk
from rag.embedding_manager import get_embedder
from rag.vector_store import AlphaVectorStore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logger configuration to monitor pipeline progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("IngestionPipeline")

def run_ingestion_pipeline(tickers: list[str]):
    """
    Executes the complete data ingestion and vectorization pipeline.
    
    :param tickers: A list of stock tickers to monitor and fetch data for (e.g., ["AAPL", "TSLA"])
    """
    logger.info("Starting Ingestion Pipeline...")

    # 1. Initialize Components
    supabase_url = os.environ.get("SUPABASE_URL")
    # Supports both your variable configurations safely
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY/ROLE_KEY not found in environment variables!")
        return

    vector_store = AlphaVectorStore(supabase_url=supabase_url, supabase_key=supabase_key)
    processor = AlphaProcessor()
    embedder = get_embedder() 
    loader = AlphaLoader(max_news_per_ticker=20, max_rss_per_feed=30) 

    # 2. Document Loading Stage (Ingestion) using native .load() method
    logger.info(f"Stage 1: Fetching multi-source documents for tickers: {tickers}...")
    try:
        all_raw_documents: list[RawDocument] = loader.load(tickers=tickers)
    except Exception as e:
        logger.error(f"Critical error during loading phase: {e}")
        return

    logger.info(f"Total raw documents successfully fetched: {len(all_raw_documents)}")

    if not all_raw_documents:
        logger.warning("No documents found to process. Terminating pipeline.")
        return

    # 3. Processing and Chunking Stage (Transformation)
    logger.info("Stage 2: Processing text and generating semantic chunks...")
    try:
        all_processed_chunks: list[ProcessedChunk] = processor.process(all_raw_documents)
    except Exception as e:
        logger.error(f"Critical error during processing phase: {e}")
        return

    logger.info(f"Processor Metrics Report: {processor.metrics.report()}")
    logger.info(f"Total semantic chunks created: {len(all_processed_chunks)}")

    if not all_processed_chunks:
        logger.warning("No new chunks remaining after deduplication filtering.")
        return

    # 4. Embedding Generation & Vector Store Upload Stage (Load)
    logger.info("Stage 3: Generating embeddings and uploading to Supabase...")
    try:
        # 1. Extract plain texts for the embedding model
        logger.info("Extracting chunk texts for embedding generation...")
        chunk_texts = [chunk.text for chunk in all_processed_chunks]
        
        # 2. Compute vectors using your custom native mini-batching method
        logger.info("Computing embedding vectors via AlphaEmbedder...")
        embeddings = embedder._encode_batch(chunk_texts)
        
        # 3. Construct record payloads according to vector_store.py requirements
        logger.info("Mapping chunks and embeddings to database schema records...")
        records_to_upsert = []
        for chunk, embedding in zip(all_processed_chunks, embeddings):
            records_to_upsert.append({
                "text": chunk.text,
                "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                "metadata": chunk.metadata
            })
        
        # 4. Upload to Supabase row by row to prevent ON CONFLICT conflict across sub-chunks
        logger.info(f"Upserting {len(records_to_upsert)} records into Supabase...")
        
        success_count = 0
        for record in records_to_upsert:
            try:
                # Safe individual upserting
                vector_store.upsert(records=[record])
                success_count += 1
            except Exception as single_exc:
                logger.warning(f"Skipped a chunk collision for URL: {record['metadata'].get('url')} | Error: {single_exc}")
        
        logger.info(f"Ingestion pipeline completed successfully. {success_count}/{len(records_to_upsert)} chunks integrated into Vector Store.")
    except Exception as e:
        logger.error(f"Error during embedding generation or database upsert: {e}")
if __name__ == "__main__":
    # Define target tickers for the financial context
    test_tickers = ["MSFT", "NVDA"]
    
    run_ingestion_pipeline(test_tickers)