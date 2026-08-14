import asyncio
from datetime import datetime, timedelta
from supabase import create_client, Client
import os


SUPABASE_URL = os.environ.get("SUPABASE_URL")

SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def check_data_sparsity(ticker: str, days: int = 14):
    print(f"--- Data Sparsity Analysis for Ticker: {ticker} ---")
    
    
    start_date = (datetime.now() - timedelta(days=days)).isoformat()
    
  
    response = supabase.table("alpha_documents") \
        .select("source_type", count="exact") \
        .eq("ticker", ticker) \
        .gte("published_at", start_date) \
        .execute()
    
    data = response.data
    total_count = len(data)
    
    print(f"Total chunks available in the last {days} days: {total_count}")
    
    
    if total_count > 0:
        distribution = {}
        for item in data:
            s_type = item.get('source_type', 'unknown')
            distribution[s_type] = distribution.get(s_type, 0) + 1
        
        print("Source distribution:")
        for source, count in distribution.items():
            print(f"  - {source}: {count} chunks")
    else:
        print("No data found in this period. The issue likely lies in the ingestion pipeline.")

if __name__ == "__main__":
    asyncio.run(check_data_sparsity("AMD"))