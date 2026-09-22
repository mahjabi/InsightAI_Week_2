# InsightAI — Multimodal Business Intelligence System
### CS 5542 Challenge 2: AI Design with Big Data, Multimodal RAG, LLMs & Agentic AI

InsightAI is an evidence-grounded AI decision support system designed to bridge the gap between large enterprise datasets and non-technical business users. It addresses the **P1 — Essential Task: Natural Language Business Data Analysis** using the Olist Brazilian E-Commerce dataset.

---

## 🌟 Key Capabilities
- **Multimodal Data Processing**: Ingests and joins structured relational tables (`orders`, `order_items`, `products`, `payments`) with unstructured qualitative text (`customer reviews` in Portuguese with English translations).
- **Separation of Responsibilities**:
  - **Database / Big Data Engine (SQL)**: Computes deterministic metrics, delivery durations, delay ratios, and pricing averages.
  - **RAG Engine**: Performs semantic retrieval across customer complaints and review texts.
  - **LLM Synthesis**: Fuses quantitative and qualitative findings into a causal narrative following:
    $$\text{Output} = \text{Answer} + \text{Evidence} + \text{Provenance} + \text{Uncertainty}$$
  - **Interactive Human Review**: Empowers human analysts to audit AI claims, flag hallucinations, and trigger automated business workflows.

---

## 🏗️ Architecture & Pipeline Flow
```
User Natural Language Question
          │
          ▼
┌────────────────────────────────────────────────────────┐
│               Separation of Responsibilities           │
├────────────────────────────┬───────────────────────────┤
│    Structured Path (SQL)   │  Unstructured Path (RAG)  │
│  - Total category orders   │  - Semantic review search │
│  - Average delivery days   │  - Complaint tag clusters │
│  - Delay percentage        │  - Negative comment text  │
│  - Price & freight values  │  - Sentiment analysis     │
└─────────────┬──────────────┴─────────────┬─────────────┘
              │                            │
              └─────────────┬──────────────┘
                            ▼
               Cross-Modal Evidence Fusion
   (Answer + Tabular Evidence + Review Citations + Provenance)
                            ▼
               Interactive Human Review Panel
   (Human auditor validates, corrects, or initiates workflows)
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/InsightAI.git
cd InsightAI
```

### 2. Install Dependencies
```bash
pip install streamlit pandas plotly
```

### 3. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure
```
InsightAI/
├── app.py                      # Main Streamlit interactive web application
├── README.md                   # Project documentation & architecture overview
├── data/
│   ├── olist_business.db       # SQLite database containing 5 linked tables & analytical views
│   ├── olist_orders.csv        # Orders table
│   ├── olist_order_items.csv   # Items & pricing table
│   ├── olist_products.csv      # Products & categories table
│   ├── olist_order_payments.csv# Payments table
│   └── olist_order_reviews.csv # Qualitative review comments & scores
```
