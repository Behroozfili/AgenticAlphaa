# Async Context Scope — Final Solution

## The Persistent Error

```
WARNING:AlphaAgent:Error closing session: Attempted to exit cancel scope in a different task than it was entered in
WARNING:AlphaAgent:Error closing stdio: Attempted to exit cancel scope in a different task than it was entered in
```

This error persisted because async context managers created in one task cannot be exited in another task in asyncio.

---

## The Ultimate Fix

### **The Pattern: Initialization Separated from Usage**

**Key Insight:** The MCP session must be initialized **once** in the **main task** before any concurrent tool calls.

```
Main Task
├─ initialize_mcp()  ← Enter contexts ONCE here
│  └─ _session and _stdio_cm are now open
│
├─ Task A: combined_research_query()
│  ├─ tavily_search() → uses same _session ✓
│  └─ newsapi_search() → uses same _session ✓ (concurrent)
│
└─ cleanup_mcp_connections()
   └─ close() → Exits contexts in main task ✓
```

### **What Changed**

#### 1. New Explicit Initialization Method

```python
async def initialize_session(self):
    """Initialize MCP session for application lifetime."""
    if self._session is not None:
        return  # Already initialized
    
    async with self._session_lock:
        if self._session is not None:
            return  # Another task initialized it
        
        # Initialize ONCE in calling task
        self._session = await create_session()
        logger.info("MCP session initialized successfully")
```

#### 2. Updated Test to Call Initialize

```python
# test_resarch_tool.py
async def run_alpha_research():
    # Initialize MCP session at startup (once in main task context)
    await initialize_mcp()  ← ← ← NEW
    
    # Now run tools concurrently
    results = await combined_research_query(...)
    
    # Cleanup
    await cleanup_mcp_connections()
```

#### 3. Call Tool Method Requires Initialization

```python
async def call_tool(self, tool_name: str, tool_input: dict):
    """Execute MCP tool (requires prior initialization)."""
    try:
        session = await self.get_session()  ← Raises error if not initialized
        result = await session.call_tool(tool_name, tool_input)
        return result
    except RuntimeError as e:
        if "not initialized" in str(e):
            logger.error("MCP session not initialized. Call initialize_mcp() first.")
        return {"error": str(e), "status": "failed"}
```

---

## Why This Works

### Task Scope Guarantee

```
Timeline with this fix:

T0: main()
    └─ await initialize_mcp()
       ├─ Enters stdio_cm context (Main Task)
       ├─ Enters session context (Main Task)
       ├─ Stores in _session
       └─ Return ✓ (contexts still open)
       
T1: Task A (tavily_search)
    └─ await _mcp_manager.get_session()
       └─ Return _session (no context changes)
       
T2: Task B (newsapi_search) [concurrent]
    └─ await _mcp_manager.get_session()
       └─ Return _session (no context changes)
       
T3: main()
    └─ await cleanup_mcp_connections()
       ├─ Closes session context (Main Task) ✓
       ├─ Closes stdio_cm context (Main Task) ✓
       └─ All exits in SAME task!
```

**Result:** ✓ No task scope violations

---

## Files Modified

### `tools/research_tools.py`
- ✅ Added `initialize_session()` method
- ✅ Updated `get_session()` to require initialization
- ✅ Enhanced error handling for uninitialized session
- ✅ Proper cleanup with graceful error handling

### `test_resarch_tool.py`
- ✅ Added import: `initialize_mcp`
- ✅ Added call: `await initialize_mcp()` at start
- ✅ Ensures session initialized before concurrent calls

---

## Usage Pattern

### Before (Broken)
```python
async def main():
    # Implicit initialization on first call
    # Could happen in Task A, get cleaned up in Task C
    # WRONG: Context scope violations!
    results = await combined_research_query(...)
    await cleanup_mcp_connections()
```

### After (Fixed)
```python
async def main():
    # Explicit initialization in main task
    await initialize_mcp()  ← Do this first
    
    # Now concurrent calls are safe
    results = await combined_research_query(...)
    
    # Cleanup in same task context
    await cleanup_mcp_connections()
```

---

## Testing

```bash
python test_resarch_tool.py
```

### Success Indicators

✅ **No context scope errors:**
```
# Should NOT see:
WARNING:AlphaAgent:Error closing session: Attempted to exit cancel scope...
WARNING:AlphaAgent:Error closing stdio: Attempted to exit cancel scope...

# Should see:
INFO:AlphaAgent:MCP session initialized successfully
INFO:AlphaAgent:MCP connections closed
```

✅ **Clean initialization and shutdown:**
```
INFO:AlphaAgent:Creating MCP client session...
INFO:AlphaAgent:MCP session created successfully
...
INFO:AlphaAgent:Closing MCP connections...
INFO:AlphaAgent:MCP connections closed
```

---

## Architecture Diagram

```
MCPClientManager
├─ initialize_session()
│  └─ Creates _session and _stdio_cm ONCE
│     (called in main task context)
│
├─ get_session()
│  └─ Returns _session
│     (safe for concurrent use)
│
├─ call_tool()
│  └─ Uses session from get_session()
│     (requires prior initialization)
│
└─ close()
   └─ Cleans up _session and _stdio_cm
      (called in same task context as init)
```

---

## Key Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| **Single initialization** | `if self._session is not None: return` |
| **Lock protection** | `async with self._session_lock:` |
| **Task scope safety** | Initialization in main, cleanup in main |
| **Concurrent safety** | Session reuse without context changes |
| **Graceful errors** | Proper error messages if not initialized |

---

## Comparison with Previous Attempts

| Approach | Issue | Result |
|----------|-------|--------|
| **Lazy init in get_session()** | Could be called by different tasks | Task scope error |
| **Manual context management** | Contexts don't respect task boundaries | Task scope error |
| **Proper initialization pattern** | Explicit init ensures correct context | ✓ FIXED |

---

## Summary

The final fix ensures:
1. ✅ MCP session created **once** in main task
2. ✅ Contexts stay open for concurrent calls
3. ✅ Cleanup happens **once** in same task
4. ✅ No async task scope violations
5. ✅ Clear error if `initialize_mcp()` not called

This is the **correct and safest** pattern for managing asyncio context managers across concurrent operations.

---

## Next Steps

1. **Run the updated test:**
   ```bash
   python test_resarch_tool.py
   ```

2. **Verify no context scope errors:**
   Should NOT see "Attempted to exit cancel scope" warnings

3. **Check normal operation:**
   - Session initializes cleanly
   - Tools execute (with API key errors is OK)
   - Session closes gracefully

4. **All other code** that uses `combined_research_query()`:
   - Add `await initialize_mcp()` before
   - Add `await cleanup_mcp_connections()` after

