# Retrieval Evaluation - Quick Start Guide

## 🚀 **Quick Start**

### **1. Run Complete Evaluation**

```bash
python evaluate_retrieval.py --all-roles --output results.json
```

This will:
- Evaluate general retrieval (all roles combined)
- Evaluate each role separately (Investor, Researcher, User)
- Generate formatted reports
- Save JSON results

### **2. View Results**

```bash
# See console output
python evaluate_retrieval.py --all-roles

# Save to file
python evaluate_retrieval.py --all-roles --output results.json

# Read results
cat retrieval_results.json
```

---

## 📊 **What Metrics Mean**

| Metric | What It Measures | Good Score | Current Performance |
|--------|------------------|------------|---------------------|
| **Hit Rate** | Did we find ANY relevant doc? | >85% | ✅ 86-100% |
| **Precision@1** | Is first result relevant? | >70% | ✅ 43-100% |
| **Precision@5** | Are top-5 results mostly relevant? | >50% | ⚠️ 20-26% |
| **Recall@5** | Did we find most relevant docs? | >60% | ⚠️ 48-54% |
| **MRR** | How early do we find the first relevant? | >0.7 | ✅ 0.6-1.0 |
| **nDCG@5** | How good is the overall ranking? | >0.8 | ✅ 0.79-1.79 |

---

## 🎯 **Key Commands**

```bash
# General retrieval only
python evaluate_retrieval.py

# All roles
python evaluate_retrieval.py --all-roles

# Specific role
python evaluate_retrieval.py --role Investor
python evaluate_retrieval.py --role Researcher
python evaluate_retrieval.py --role User

# Custom queries file
python evaluate_retrieval.py --queries my_queries.json

# Skip general (roles only)
python evaluate_retrieval.py --skip-general --all-roles

# Save results
python evaluate_retrieval.py --output my_results.json
```

---

## 📝 **Adding Test Queries**

Edit `test_queries.json`:

```json
[
  {
    "query": "Your test query here",
    "relevant_docs": [
      "knowledge_base\\document1.pdf",
      "knowledge_base\\document2.txt"
    ],
    "relevance_scores": {
      "knowledge_base\\document1.pdf": 1.0,
      "knowledge_base\\document2.txt": 0.8
    },
    "role": "Investor"
  }
]
```

**Important**:
- Use `\\` or `/` for file paths
- Relevance scores: 0.0 (irrelevant) to 1.0 (highly relevant)
- Role: "Investor", "Researcher", or "User"

---

## 🔍 **Example Output**

```
======================================================================
RETRIEVAL QUALITY EVALUATION - Investor
======================================================================

Total Queries Evaluated: 7

[METRICS SUMMARY]
----------------------------------------------------------------------
Hit Rate:              0.857 (85.7%)      <- Did we find something? YES
Mean Reciprocal Rank:  0.600              <- First relevant at position 1.67

Precision@k:
  @1:  0.429 (42.9%)                      <- First result is relevant 43% of time
  @3:  0.381 (38.1%)                      <- Top 3: 38% are relevant
  @5:  0.257 (25.7%)                      <- Top 5: 26% are relevant

Recall@k:
  @1:  0.167 (16.7%)                      <- Found 17% of relevant docs in top 1
  @3:  0.429 (42.9%)                      <- Found 43% of relevant docs in top 3
  @5:  0.476 (47.6%)                      <- Found 48% of relevant docs in top 5

nDCG@k (Gold Standard):
  @1:  0.429                              <- Ranking quality at 1
  @3:  0.587                              <- Ranking quality at 3
  @5:  0.787                              <- Ranking quality at 5
======================================================================
```

---

## 🎓 **Understanding the Results**

### **Good Results** ✅
- Hit Rate >85%: We're finding relevant documents
- Precision@1 >70%: First result is usually relevant
- MRR >0.7: Relevant documents appear early
- nDCG@5 >0.8: Good overall ranking quality

### **Needs Improvement** ⚠️
- Precision@5 <50%: Too much noise in later results
- Recall@5 <60%: Missing some relevant documents
- MRR <0.5: Relevant documents appear too late

### **Example Interpretation**

**Researcher Agent**: 
- Hit Rate: 100% ✅ (Perfect - always finds relevant docs)
- Precision@1: 100% ✅ (Perfect - first result always relevant!)
- MRR: 1.0 ✅ (Perfect - first result is always first!)
- **Conclusion**: Outstanding performance!

**Investor Agent**:
- Hit Rate: 85.7% ✅ (Good - usually finds relevant docs)
- Precision@1: 42.9% ⚠️ (Moderate - first result relevant less than half the time)
- MRR: 0.6 ⚠️ (Moderate - relevant docs appear around position 2)
- **Conclusion**: Needs improvement in query ranking

---

## 🔧 **Troubleshooting**

### **Issue: No results found**

```bash
# Check knowledge base initialization
python -c "from multi_agent_system_simple import RAGKnowledgeBase; kb = RAGKnowledgeBase(); print(kb.collection.count())"
```

Should show >1000 chunks.

### **Issue: All queries return 0%**

Check that file paths in `test_queries.json` match actual files:
```bash
ls knowledge_base/
```

### **Issue: Encoding errors**

Already fixed! Windows console now handled properly.

---

## 📚 **Advanced Usage**

### **Compare Two Runs**

```bash
# Run 1
python evaluate_retrieval.py --all-roles --output baseline.json

# ... make changes ...

# Run 2
python evaluate_retrieval.py --all-roles --output improved.json

# Compare
diff baseline.json improved.json
```

### **Focused Testing**

```bash
# Test only Investor role
python evaluate_retrieval.py --role Investor --skip-general

# Test custom queries
python evaluate_retrieval.py --queries my_test_set.json --role Researcher
```

---

## 💡 **Tips**

1. **Start with small test sets** (5-10 queries) for quick iteration
2. **Add queries gradually** to build comprehensive evaluation set
3. **Focus on your use case** - if you're building a finance app, add more Investor queries
4. **Use relevance scores** to weight more important documents
5. **Compare roles** to understand which agent works best for different query types

---

## 📖 **Further Reading**

- `RETRIEVAL_EVALUATION.md` - Full evaluation report with analysis
- `evaluation.py` - Source code for all metrics
- `evaluate_retrieval.py` - Evaluation script implementation
- `test_queries.json` - Example query set

---

**Questions?** Check the full documentation in `RETRIEVAL_EVALUATION.md`


