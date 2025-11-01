# Quick Start Guide - Fix API Connection

## 🔴 **Current Issue**

Your Streamlit UI shows "API Not Available" because the API server needs to be running.

---

## ✅ **Solution: Start API Server**

### **Method 1: Double-Click Batch File** (Easiest)

1. In Windows Explorer, navigate to: `D:\multiagent_system`
2. **Double-click**: `start_api_server.bat`
3. A terminal window will open showing the server starting
4. Wait for: `INFO:     Application startup complete.`
5. **Keep this window open** (don't close it!)
6. Go back to your browser and **refresh** the Streamlit page

### **Method 2: Manual PowerShell** 

Open a **NEW** PowerShell window and run:

```powershell
cd D:\multiagent_system
.\.venv\Scripts\Activate.ps1
python server_simple.py
```

Wait for the message: `INFO:     Application startup complete.`

---

## ✅ **After Starting**

1. **Refresh your browser** (F5 or click refresh button)
2. The red "API Not Available" should change to:
   - ✅ **Green**: "API Connected"
3. You should see:
   - System info in sidebar
   - All tabs working
   - Ready to submit startup ideas!

---

## 🔍 **Verify It's Working**

Once the API starts, you'll see in the terminal:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## ⚠️ **Important Notes**

- **Keep the API server terminal open** while using the app
- The first startup may take 2-5 minutes (initializing knowledge base)
- After first run, it starts faster

---

## 🎯 **Quick Test**

Once both are running:
1. Go to Streamlit UI: http://localhost:8501
2. You should see: ✅ API Connected (green)
3. Try submitting a startup idea!

---

**Status**: API server needs to be started manually  
**Fix**: Run `start_api_server.bat` or use Method 2 above

