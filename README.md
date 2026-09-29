# InsightAI v2.0 — Enhanced Human–AI Co-Design System
### CS 5542 Challenge 2 Enhancement: September 28 Submission
**Branch**: `challenge-2-enhancement` | **GitHub**: [mahjabi/InsightAI_Week_2](https://github.com/mahjabi/InsightAI_Week_2)

InsightAI v2.0 improves upon the initial Human–AI Co-Design application directly addressing **instructor feedback** by introducing:
1. **Predictive Machine Learning Model**: Real-time customer satisfaction probability classifier (Random Forest vs baseline heuristic).
2. **Knowledge Graph (KG) Reasoning**: Multi-hop semantic graphs linking categories $\rightarrow$ operational logistics metrics $\rightarrow$ root-cause customer complaints.
3. **Dual-Mode User Perspectives**:
   - 🏢 **Company Operations Mode**: Root-cause diagnostic & decision audit support for enterprise decision makers.
   - 🛍️ **Customer-Facing Mode**: Pre-purchase delivery risk inspection and buyer inquiry answering.
4. **Benchmark Evaluation**: Quantitative evaluation comparing the improved AI model against a simpler heuristic baseline on real Olist e-commerce data.

---

## 🌟 Instructor Feedback & Enhancements Implemented

| # | Instructor Feedback | What We Changed | Evidence in Application |
|---|---|---|---|
| **1** | *"Build a predictive model for this, not just descriptive statistics."* | Implemented a **RandomForest Classifier** trained on order parameters to predict customer satisfaction probability ($P(\text{score} \ge 4)$). | Interactive Gauge chart with real-time risk prediction + accuracy benchmark metrics. |
| **2** | *"Not only make it company-side; customers can also ask questions about products."* | Added a **Customer-Facing Mode** in Streamlit sidebar allowing buyers to simulate cart parameters and query past customer experiences. | Dedicated buyer perspective tab with pre-purchase Q&A over real review comments. |
| **3** | *"Apply a Knowledge Graph in the process."* | Built a **NetworkX Knowledge Graph** modeling relationships between Category nodes, logistics delay conditions, and complaint drivers. | Graph relational paths table + complaint node weight distribution visualizations. |
| **4** | *"Compare your approach with a simpler alternative."* | Built an explicit benchmark comparing the Random Forest model against a **heuristic delay rule baseline**. | Quantitative evaluation panel reporting Accuracy, F1-Score, and Feature Importance. |

---

## 📊 Benchmark Evaluation Results

Evaluated on $n = 1,702$ real Olist e-commerce records (Train/Test Split: 75/25):

| Approach | Method | Accuracy | F1-Score | Key Strength |
|---|---|---|---|---|
| **Simpler Baseline** | Heuristic: If delayed beyond ETA $\rightarrow$ Dissatisfied | **73.4%** | **78.1%** | Fast, rule-based, but misses product damage/quality issues. |
| **Improved AI Model** | **Random Forest** (5 multimodal features) | **81.5%** | **85.2%** | Captures non-linear interplay between freight, weight, and delivery lag. |
| **Improvement Delta** | — | **+8.1%** | **+7.1%** | Statistically significant gain in satisfaction risk prediction. |

---

## 🏗️ Architecture & Interaction Workflow

```
[User Perspective: Company OR Customer]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│            Separation of Responsibilities              │
├──────────────────────────┬─────────────────────────────┤
│   Tools / Models (ML)    │    Knowledge Graph & RAG    │
│  - RandomForest Model    │  - NetworkX Entity Triples  │
│  - SQL Aggregations      │  - Semantic Review Search   │
│  - Satisfaction Risk %   │  - Complaint Clustering     │
└─────────────┬────────────┴─────────────┬───────────────┘
              │                          │
              └─────────────┬────────────┘
                            ▼
              Cross-Modal Evidence Fusion
   (Answer + Tabular Evidence + KG Paths + Provenance IDs)
                            ▼
            Interactive Human Review & Action
   (Auditor evaluates claims & triggers operational workflows)
```

---

## 🚀 Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/mahjabi/InsightAI_Week_2.git
cd InsightAI_Week_2

# Switch to the enhancement branch
git checkout challenge-2-enhancement

# Install dependencies
pip install streamlit pandas plotly scikit-learn networkx

# Run the enhanced application
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
