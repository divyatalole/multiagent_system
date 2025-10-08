# StartupAI - Multi-Agent Thinking Partner System

A sophisticated **Multi-Agent Thinking Partner** that bridges the gap between research and practical application by leveraging advanced AI technologies for comprehensive startup evaluation.

## 🚀 **System Architecture**

### **Core Components**
- **Role-Specific Agents**: Investor, Researcher, and User agents powered by local LLMs
- **RAG-Based Architecture**: Dynamic information retrieval with role-relevant knowledge
- **Model Context Protocol (MCP)**: Ensures agents are aware of each other's feedback
- **Multi-Criteria Decision Making (MCDM)**: Synthesizes differing opinions into actionable insights

### **Technology Stack**
- **Backend**: Python with FastAPI, LangChain, and local LLM integration
- **Frontend**: Modern HTML5, CSS3, JavaScript with responsive design
- **AI Models**: Local Mistral 7B LLM with ctransformers
- **Vector Database**: ChromaDB with sentence transformers for embeddings
- **Decision Making**: scikit-criteria for MCDM algorithms

## 🎯 **Key Features**

### **Multi-Agent AI Analysis**
- **Investor Agent**: Financial & market analysis, investment potential, risk assessment
- **Researcher Agent**: Technical feasibility, market research, competitive landscape
- **User Agent**: User experience evaluation, adoption potential, user value proposition

### **Advanced AI Capabilities**
- **RAG Integration**: Dynamic knowledge retrieval from startup and business domain databases
- **Context Awareness**: Agents understand and build upon each other's analyses
- **Confidence Scoring**: Each analysis includes confidence metrics and reasoning
- **Consensus Building**: MCDM algorithms synthesize multiple perspectives

### **Comprehensive Evaluation Process**
1. **Startup Submission**: Detailed form for startup idea submission
2. **Individual Analysis**: Each AI agent provides independent evaluation with RAG-enhanced insights
3. **Collaborative Discussion**: MCP-enabled agents discuss findings and build consensus
4. **Final Recommendation**: MCDM-based consolidated assessment with actionable insights
5. **Results Summary**: Visual metrics and downloadable reports

## 🛠️ **Installation & Setup**

### **Prerequisites**
- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended for LLM)
- 4GB+ free disk space for models

### **Quick Start**
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multiagent_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model file**
   ```
   model/mistral-7b-instruct-v0.2.Q4_K_M.gguf
   ```

4. **Test the system (CLI)**
   ```bash
   python cli.py
   ```

5. **Run the API server**
   ```bash
   python server.py
   ```

6. **Open the frontend**
   ```
   open index.html
   ```

## 🗂️ Project Layout

- `cli.py`: CLI runner for the full multi-agent system
- `server.py`: FastAPI server exposing the API
- `multi_agent_system.py`: Full-featured system (RAG, MCP, MCDM)
- `multi_agent_system_simple.py`: Lightweight PDF/text analyzer + simple agents
- `interactive_pdf_analyzer.py`: CLI demo on top of the simplified system
- `investor_ollama.py`: Ollama-based investor JSON analysis
- `knowledge_base/`: Text and PDF documents
- `model/`: Local model files (GGUF)
- `tests/`: Automated tests (`tests/test_pdf.py`)
- Frontend: `index.html`, `script.js`, `styles.css`

## 🏗️ **System Architecture Details**

### **RAG Knowledge Base**
- **Domain Knowledge**: Startup funding, market analysis, technical feasibility, UX design
- **Document Support**: Text files (.txt) and PDF documents (.pdf)
- **Dynamic Retrieval**: Role-specific information retrieval for each agent
- **Vector Embeddings**: Sentence transformers for semantic search
- **Knowledge Chunks**: Optimized document splitting for context-aware retrieval
- **PDF Processing**: Automatic text extraction and metadata handling

### **Model Context Protocol (MCP)**
- **Agent Communication**: Structured information sharing between agents
- **Context Awareness**: Each agent understands others' perspectives
- **Feedback Integration**: Agents build upon previous analyses
- **Consensus Building**: Collaborative decision-making process

### **Multi-Criteria Decision Making (MCDM)**
- **Criteria Weighting**: Balanced evaluation across multiple dimensions
- **Normalization**: Standardized scoring across different metrics
- **Ranking Algorithms**: TOPSIS and weighted sum methods
- **Consensus Generation**: Unified recommendations from diverse perspectives

## 📊 **Evaluation Metrics**

### **Agent-Specific Metrics**
- **Investment Potential**: 1-10 scale with risk assessment
- **Technical Feasibility**: Technology maturity and implementation complexity
- **User Experience**: Adoption potential and UX quality scores
- **Confidence Scores**: Analysis reliability indicators

### **Consensus Metrics**
- **Overall Score**: Weighted combination of all agent scores
- **Risk Level**: Low/Medium/High classification
- **Market Potential**: Market size and growth opportunity assessment
- **Feasibility**: Technical and operational feasibility rating

## 🔧 **API Endpoints**

### **Core Endpoints**
- `GET /health` - Health check
- `POST /analyze` - Quick idea analysis (persona, idea, top_k)
- `POST /evaluate` - Full startup evaluation
- `GET /agents` - List available agents
- `GET /knowledge/files` - List KB files
- `POST /knowledge/upload` - Upload file to KB
- `GET /evaluations` - List cached evaluations

### **Response Format**
```json
{
  "status": "success",
  "data": {
    "startup_data": {...},
    "agent_analyses": [...],
    "consensus_decision": {...},
    "evaluation_timestamp": "..."
  }
}
```

## 🎨 **Frontend Features**

### **Modern Web Interface**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Live agent analysis progress
- **Interactive Elements**: Smooth animations and transitions
- **API Integration**: Connects to real multi-agent backend

### **User Experience**
- **Progressive Disclosure**: Information revealed gradually
- **Loading States**: Realistic AI analysis simulation
- **Error Handling**: Graceful fallback to simulation mode
- **Report Generation**: Downloadable evaluation reports

## 🚀 **Advanced Usage**

### **Custom Agent Development**
1. **Extend Agent Class**: Inherit from base Agent class
2. **Implement Analysis**: Define role-specific evaluation logic
3. **Add to System**: Register new agent in MultiAgentSystem
4. **Update MCDM**: Include new agent in decision-making process

### **Knowledge Base Expansion**
1. **Add Text Documents**: Place new knowledge files in knowledge_base/
2. **Add PDF Documents**: Use the provided tools to add PDFs
3. **Update Embeddings**: Reinitialize vector store
4. **Role Context**: Define role-specific retrieval patterns
5. **Validation**: Test retrieval quality and relevance

### **PDF Document Management**
1. **Add PDFs**: Use `python add_pdf_to_kb.py "path/to/document.pdf"`
2. **Batch File**: Windows users can use `add_pdf.bat "path/to/document.pdf"`
3. **List Contents**: Check current knowledge base with `python add_pdf_to_kb.py --list`
4. **Automatic Processing**: PDFs are automatically extracted and indexed
5. **Metadata Handling**: Source file, page numbers, and file type are preserved

### **Demos & Examples**
- CLI demo (PDF KB, simple agents): `python interactive_pdf_analyzer.py`
- Ollama investor JSON: `python investor_ollama.py`

### **MCDM Customization**
1. **Criteria Weights**: Adjust importance of different factors
2. **Scoring Methods**: Implement custom normalization algorithms
3. **Consensus Rules**: Define how agents reach agreement
4. **Thresholds**: Set minimum confidence and score requirements

## 🌟 **Future Enhancements**

### **Planned Features**
- **Real-time Collaboration**: Multiple users evaluating same startup
- **Market Data Integration**: Live market trends and competitor analysis
- **Advanced NLP**: Better text analysis and metric extraction
- **Model Fine-tuning**: Domain-specific LLM training

### **Scalability Improvements**
- **Distributed Agents**: Multi-server agent deployment
- **Caching Layer**: Redis-based result caching
- **Load Balancing**: Multiple LLM instance support
- **Async Processing**: Improved concurrent analysis

## 📱 **Browser Support**

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### **Development Guidelines**
- Follow PEP 8 coding standards
- Add type hints for all functions
- Include comprehensive docstrings
- Write unit tests for new features
- Update documentation as needed

## 📄 **License**

This project is open source and available under the MIT License.

## 🆘 **Support & Troubleshooting**

### **Common Issues**
1. **Model Loading Errors**: Ensure sufficient RAM and correct model path
2. **Import Errors**: Install all dependencies from requirements.txt
3. **API Connection**: Check if FastAPI server is running on port 8000
4. **Performance Issues**: Adjust LLM parameters for your hardware

### **Getting Help**
- Open an issue in the repository
- Check the troubleshooting section
- Review system logs for error details
- Ensure all dependencies are properly installed

---

**StartupAI Multi-Agent Thinking Partner** - Empowering entrepreneurs with AI-powered startup evaluation since 2024.

*Built with cutting-edge AI technology including RAG, MCP, and MCDM for comprehensive startup analysis.*

