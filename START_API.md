# How to Start the API Server

## Quick Fix

The Streamlit UI is running but needs the API server to be started separately.

### Option 1: Use PowerShell (Current Terminal)

```powershell
cd D:\multiagent_system
.\.venv\Scripts\Activate.ps1
python server_simple.py
```

### Option 2: Use New Terminal Window

1. Open a **NEW PowerShell/Command Prompt window**
2. Navigate to project:
   ```powershell
   cd D:\multiagent_system
   ```
3. Activate virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. Start API server:
   ```powershell
   python server_simple.py
   ```

### What You Should See

When the API server starts successfully, you'll see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### After Starting

1. Go back to your browser (Streamlit UI)
2. Click the **refresh button** or reload the page
3. The red "API Not Available" should change to green "✅ API Connected"

### Verify It's Working

Once started, you should see:
- ✅ API Connected (green box in sidebar)
- All tabs working
- Ability to submit startup ideas

---

**Note**: Keep the API server terminal window open while using the app!

