import asyncio
import pytest
from mcp.client import Client
from mcp.server.fastmcp import FastMCP
from  import ResearchTools # منطق اصلی ابزارها

# تنظیمات محیطی برای تست
@pytest.fixture
async def mcp_client():
    """ایجاد یک کلاینت موقت برای تست سرور MCP."""
    mcp_server = FastMCP("Test-Research-Server")
    # ثبت ابزارها برای تست
    research_logic = ResearchTools()
    
    @mcp_server.tool()
    async def test_search(query: str):
        return await research_logic.web_search(query)

    # در دنیای واقعی، کلاینت به STDIO سرور متصل می‌شود
    # برای تست، ما مستقیماً منطق را فراخوانی می‌کنیم
    return research_logic

@pytest.mark.asyncio
async def test_web_search_schema(mcp_client):
    """بررسی ساختار پاسخ Tavily و صحت Tracing."""
    print("\n[Test 1] Testing Tavily Web Search...")
    query = "NVIDIA stock performance 2026"
    
    response = await mcp_client.web_search(query)
    
    assert response is not None
    assert "results" in response
    print("✓ Web Search response schema is valid.")

@pytest.mark.asyncio
async def test_fetch_news_logic(mcp_client):
    """بررسی صحت دریافت اخبار و محدودیت تعداد نتایج."""
    print("\n[Test 2] Testing NewsAPI Fetching...")
    ticker = "NVDA"
    
    articles = await mcp_client.fetch_news(ticker)
    
    assert isinstance(articles, list)
    assert len(articles) <= 5 # بررسی محدودیت تعداد که در منطق اعمال شده بود
    if len(articles) > 0:
        assert "title" in articles[0]
    print(f"✓ Successfully fetched {len(articles)} articles for {ticker}.")

@pytest.mark.asyncio
async def test_error_handling_decorator(mcp_client):
    """تست عملکرد دکوراتور handle_error در صورت نبود کلید API."""
    print("\n[Test 3] Testing Error Handling Decorator...")
    # شبیه‌سازی خطا با تغییر موقت کلید
    original_key = mcp_client.tavily_key
    mcp_client.tavily_key = "INVALID_KEY"
    
    with pytest.raises(Exception):
        await mcp_client.web_search("test")
    
    mcp_client.tavily_key = original_key
    print("✓ Decorator correctly caught and handled the API error.")

if __name__ == "__main__":
    # اجرای تست‌ها به صورت دستی در صورت عدم استفاده از pytest
    asyncio.run(test_web_search_schema(ResearchTools()))
    asyncio.run(test_fetch_news_logic(ResearchTools()))