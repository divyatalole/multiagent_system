# Streamlit Web Interface - User Guide

## 🚀 **StartupAI Web App**

Modern, user-friendly web interface for startup evaluation using Streamlit.

---

## 📋 **Quick Start**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Start API Server**

```bash
python server_simple.py
```

Wait for: `Application startup complete`

### **3. Start Streamlit App**

In a new terminal:

```bash
streamlit run streamlit_app.py
```

### **4. Open Browser**

App opens automatically at: `http://localhost:8501`

---

## 🎨 **Features**

### **1. New Analysis Tab** 📝

**Purpose**: Submit startup ideas for initial evaluation

**Usage**:
1. Enter startup description
2. Adjust retrieval settings
3. Click "Analyze Startup"
4. View comprehensive results

**Input**:
- Startup description (text area)
- Number of documents (slider: 3-10)

**Output**:
- Overall score and metrics
- Investor analysis
- Researcher insights
- User perspective
- Supporting documents

---

### **2. Conversation Tab** 💬

**Purpose**: Ask follow-up questions interactively

**Usage**:
1. Run initial analysis first
2. Enter follow-up question
3. Select target agent (optional)
4. View detailed response

**Features**:
- Context-aware responses
- Agent-specific answers
- Multi-agent perspectives

**Example Questions**:
- "What are the main risks?"
- "How big is the market opportunity?"
- "What's the user experience like?"
- "Can you elaborate on the technical feasibility?"

---

### **3. Session History Tab** 📊

**Purpose**: View and manage past analyses

**Features**:
- All analyses in current session
- Timestamp for each analysis
- Quick access to previous results
- Clear history option

---

## 🎯 **User Journey**

### **Step 1: Submit Idea**

```
User enters startup description
    ↓
Clicks "Analyze Startup"
    ↓
AI agents analyze in parallel
    ↓
Results displayed
```

### **Step 2: Explore Results**

```
View overall assessment
    ↓
Read each agent's insights
    ↓
Review recommendations
    ↓
Check supporting documents
```

### **Step 3: Deep Dive**

```
Switch to Conversation tab
    ↓
Ask specific questions
    ↓
Get detailed responses
    ↓
Continue dialogue
```

---

## 💻 **Interface Components**

### **Sidebar**
- ✅ API connection status
- 📊 System statistics
- ℹ️ About information

### **Main Area**

**Tab 1: New Analysis**
- Input form
- Analysis button
- Results dashboard
- Agent cards

**Tab 2: Conversation**
- Follow-up form
- Agent selector
- Interactive responses
- Chat-like interface

**Tab 3: History**
- Timeline view
- Expandable entries
- Clear button

---

## 🎨 **Visual Design**

### **Color Coding**

**Investor Agent** 💰
- Blue theme
- Financial focus
- ROI metrics

**Researcher Agent** 🔬
- Green theme
- Data-driven
- Market analysis

**User Agent** 👥
- Pink theme
- UX perspective
- Adoption insights

---

## 🔧 **Configuration**

### **Change API URL**

Edit `streamlit_app.py`:

```python
API_BASE_URL = "http://localhost:8000"  # Change if needed
```

### **Adjust Timeouts**

```python
timeout=60  # Seconds
```

### **Customize Styling**

Edit CSS in `st.markdown("""<style>...</style>""")`

---

## 🐛 **Troubleshooting**

### **Issue: "API Not Available"**

**Solution**:
1. Check if API server is running
2. Verify URL: `http://localhost:8000`
3. Test: `curl http://localhost:8000/health`

### **Issue: "Analysis Timeout"**

**Solution**:
1. Increase timeout in code
2. Check API server logs
3. Reduce max_results slider

### **Issue: "No Results"**

**Solution**:
1. Verify knowledge base is loaded
2. Check API health endpoint
3. Review startup description

---

## 📊 **Performance**

**Expected Speeds**:
- Initial analysis: 10-30 seconds
- Follow-up questions: 5-15 seconds
- Document retrieval: 1-3 seconds

**Optimization Tips**:
- Use fewer documents (3-5) for speed
- Start with broad questions
- Use specific follow-ups for details

---

## ✅ **Testing**

### **Test Workflow**

1. Start both servers
2. Open Streamlit app
3. Submit test startup
4. Verify results
5. Ask follow-up
6. Check history

### **Test Data**

```python
# Good example
startup_description = """
AI-powered personal finance app that helps millennials save money 
automatically based on spending patterns and financial goals. Features 
include automated budget allocation, predictive spending analysis, 
and goal-based investment recommendations.
"""
```

---

## 🚀 **Deployment**

### **Local Development**
```bash
streamlit run streamlit_app.py
```

### **Production**

**Option 1: Streamlit Cloud**
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

**Option 2: Docker**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

**Option 3: Custom Server**
- Use Streamlit in server mode
- Configure reverse proxy
- Add SSL certificates

---

## 📚 **API Integration**

### **Endpoints Used**

1. **GET /health** - Health check
2. **GET /status** - System status
3. **GET /agents** - List agents
4. **POST /analyze** - Run analysis
5. **POST /analyze/followup** - Follow-up questions

### **Request/Response**

See `server_simple.py` for API details

---

## ✅ **Feature Checklist**

- [x] Startup submission form
- [x] Multi-agent analysis display
- [x] Follow-up question interface
- [x] Conversation history
- [x] Agent-specific routing
- [x] Real-time status updates
- [x] Error handling
- [x] Responsive design
- [x] API health checks
- [x] Session management

---

## 🎉 **Success Criteria**

✅ **Functional**: All features working  
✅ **User-Friendly**: Intuitive interface  
✅ **Fast**: < 30s initial analysis  
✅ **Reliable**: Error handling robust  
✅ **Modern**: Clean, professional design  

---

**Status**: ✅ **COMPLETE AND READY**  
**URL**: `http://localhost:8501`  
**Dependencies**: API server running  
**Quality**: Production Ready ✅

---

*Built with Streamlit 1.29*  
*FastAPI backend*  
*Multi-agent AI system*

