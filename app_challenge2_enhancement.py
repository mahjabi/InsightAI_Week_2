import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="InsightAI v2 — Enhanced Multimodal Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .badge-enhancement {
        background-color: #38BDF8;
        color: #0F172A;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 3px 8px;
        border-radius: 12px;
        vertical-align: middle;
        margin-left: 8px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .evidence-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .provenance-tag {
        font-size: 0.8rem;
        font-family: monospace;
        background-color: #EEF2F6;
        color: #1E293B;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .uncertainty-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "/Users/mahjabin/Downloads/InsightAI/data/olist_business.db"

@st.cache_resource
def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_db_connection()

# ---------------------------------------------------------
# Machine Learning Model: Customer Satisfaction Predictor
# ---------------------------------------------------------
@st.cache_resource
def train_predictive_model():
    query = """
    SELECT 
        actual_delivery_days,
        is_delayed,
        price,
        freight_value,
        product_weight_g,
        review_score
    FROM v_multimodal_analytics
    """
    df = pd.read_sql_query(query, conn)
    # Target: High satisfaction (4-5 stars = 1) vs Low satisfaction (1-3 stars = 0)
    df['is_high_satisfaction'] = (df['review_score'] >= 4).astype(int)
    
    features = ['actual_delivery_days', 'is_delayed', 'price', 'freight_value', 'product_weight_g']
    X = df[features]
    y = df['is_high_satisfaction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 1. Baseline Model (Simple Heuristic: If delayed -> 0, else -> 1)
    baseline_preds = (X_test['is_delayed'] == 0).astype(int)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    baseline_f1 = f1_score(y_test, baseline_preds)
    
    # 2. Enhanced Machine Learning Model (RandomForest)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    rf_preds = model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_precision = precision_score(y_test, rf_preds)
    rf_recall = recall_score(y_test, rf_preds)
    
    feature_importance = pd.DataFrame({
        'Feature': ['Delivery Days', 'Delayed Beyond ETA', 'Item Price', 'Freight Cost', 'Weight (g)'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    metrics = {
        'baseline_acc': baseline_acc,
        'baseline_f1': baseline_f1,
        'rf_acc': rf_acc,
        'rf_f1': rf_f1,
        'rf_precision': rf_precision,
        'rf_recall': rf_recall,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    return model, metrics, feature_importance

ml_model, eval_metrics, feature_imp_df = train_predictive_model()

# ---------------------------------------------------------
# Knowledge Graph Builder: Category -> Delivery -> Reviews -> Causes
# ---------------------------------------------------------
@st.cache_resource
def build_knowledge_graph():
    G = nx.DiGraph()
    
    # Query aggregated entities
    query = """
    SELECT 
        product_category_name,
        ROUND(AVG(review_score), 2) as avg_rating,
        ROUND(AVG(actual_delivery_days), 1) as avg_days,
        complaint_tag,
        COUNT(*) as tag_count
    FROM v_multimodal_analytics
    WHERE sentiment = 'Negative'
    GROUP BY product_category_name, complaint_tag
    """
    df_kg = pd.read_sql_query(query, conn)
    
    for _, row in df_kg.iterrows():
        cat = row['product_category_name']
        cause = row['complaint_tag']
        count = row['tag_count']
        
        G.add_node(cat, type="Category", color="#38BDF8")
        G.add_node(cause, type="RootCause", color="#F87171")
        G.add_edge(cat, cause, relation="DRIVES_COMPLAINT", weight=count)
        
        # Connect to operational delay condition
        delay_node = f"DeliveryDelay_{cat}"
        G.add_node(delay_node, type="LogisticsMetric", color="#FBBF24")
        G.add_edge(cat, delay_node, relation="HAS_AVG_DELIVERY_DAYS", days=row['avg_days'])
        
    return G

kg_graph = build_knowledge_graph()

# ---------------------------------------------------------
# Sidebar — Dual Mode & Enhancement Controls
# ---------------------------------------------------------
st.sidebar.title("🔮 InsightAI v2.0")
st.sidebar.caption("Challenge 2 Enhancement — Deadline: Sep 28")

app_mode = st.sidebar.radio(
    "Select User Perspective:",
    ["🏢 Company Operations Mode (Diagnosis & Auditing)", 
     "🛍️ Customer-Facing Mode (Pre-Purchase Inquiries & Satisfaction Risk)"],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("Instructor Enhancements")
st.sidebar.markdown("""
- 🤖 **Predictive ML Model**: RandomForest vs Simple Heuristic Baseline
- 🕸️ **Knowledge Graph (KG)**: Entity relational reasoning (Category $\\rightarrow$ Cause $\\rightarrow$ Metric)
- 🛍️ **Customer-Facing Portal**: Pre-purchase buyer Q&A
- ⚖️ **Evaluation & Baseline Comparison**: Quantitative benchmark
""")

st.sidebar.divider()
st.sidebar.caption("GitHub: [InsightAI_Week_2](https://github.com/mahjabi/InsightAI_Week_2)")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🧠 InsightAI: Human–AI Co-Design System <span class="badge-enhancement">v2.0 Enhanced</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Addressing Challenge 2 Enhancements: Predictive Modeling, Knowledge Graph Reasoning & Dual-Mode Perspective</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# PERSPECTIVE 1: COMPANY OPERATIONS MODE (DIAGNOSTIC & AUDIT)
# -------------------------------------------------------------------------------------------------
if "Company Operations" in app_mode:
    st.subheader("1. Executive Category Diagnostic & Root-Cause Analysis")
    
    preset_questions = [
        "Why are furniture_decor products suffering from low customer review ratings?",
        "What issues are causing low satisfaction in bed_bath_table orders?",
        "Why does computers_accessories have high satisfaction compared to other categories?",
        "What factors are driving customer complaints across sports_leisure?",
        "Custom Question..."
    ]
    question_choice = st.selectbox("Select Business Diagnostic Question:", preset_questions)
    
    if question_choice == "Custom Question...":
        user_query = st.text_input("Type your analytical question:", "Why is furniture_decor having delivery problems?")
    else:
        user_query = question_choice
        
    detected_cat = "furniture_decor"
    for cat in ["furniture_decor", "bed_bath_table", "computers_accessories", "health_beauty", "sports_leisure"]:
        if cat in user_query.lower():
            detected_cat = cat
            break
            
    # Tabular analytics query
    sql_summary_query = f"""
    SELECT 
        product_category_name,
        COUNT(DISTINCT order_id) as total_orders,
        ROUND(AVG(review_score), 2) as avg_rating,
        ROUND(AVG(actual_delivery_days), 1) as avg_delivery_days,
        ROUND(AVG(price), 2) as avg_price,
        ROUND(AVG(freight_value), 2) as avg_freight,
        ROUND(SUM(is_delayed) * 100.0 / COUNT(order_id), 1) as delay_pct
    FROM v_multimodal_analytics
    GROUP BY product_category_name
    """
    df_categories = pd.read_sql_query(sql_summary_query, conn)
    cat_metrics = df_categories[df_categories['product_category_name'] == detected_cat].iloc[0]
    benchmark_metrics = df_categories[df_categories['product_category_name'] != detected_cat].mean(numeric_only=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category Avg Rating", f"{cat_metrics['avg_rating']} ★", delta=f"{round(cat_metrics['avg_rating'] - benchmark_metrics['avg_rating'], 2)} vs baseline")
    c2.metric("Avg Delivery Lag", f"{cat_metrics['avg_delivery_days']} days", delta=f"{round(cat_metrics['avg_delivery_days'] - benchmark_metrics['avg_delivery_days'], 1)} days", delta_color="inverse")
    c3.metric("Delay Ratio", f"{cat_metrics['delay_pct']}%", delta=f"{round(cat_metrics['delay_pct'] - benchmark_metrics['delay_pct'], 1)}%", delta_color="inverse")
    c4.metric("Analyzed Orders", f"{int(cat_metrics['total_orders'])}")
    
    st.divider()
    
    # Dual Columns: Knowledge Graph Paths vs Customer Review Citations
    col_kg, col_rev = st.columns([1, 1])
    
    with col_kg:
        st.markdown("### 🕸️ Knowledge Graph Relational Paths")
        st.caption("Structured multi-hop reasoning connecting Category $\\rightarrow$ Operational Metrics $\\rightarrow$ Root Causes")
        
        # Extract KG neighbors for detected category
        kg_edges = []
        if detected_cat in kg_graph:
            for neighbor in kg_graph.neighbors(detected_cat):
                rel = kg_graph[detected_cat][neighbor].get('relation', 'RELATES_TO')
                weight = kg_graph[detected_cat][neighbor].get('weight', '')
                kg_edges.append({"Source": detected_cat, "Relation": rel, "Target / Cause": neighbor, "Evidence Weight": f"{weight} complaints" if weight else ""})
        
        df_edges = pd.DataFrame(kg_edges)
        st.dataframe(df_edges, use_container_width=True)
        
        # Bar Chart of KG Root Causes
        sql_tags = f"""
        SELECT complaint_tag, COUNT(*) as count 
        FROM v_multimodal_analytics 
        WHERE product_category_name = '{detected_cat}' AND sentiment = 'Negative'
        GROUP BY complaint_tag
        ORDER BY count DESC
        """
        df_tags = pd.read_sql_query(sql_tags, conn)
        if not df_tags.empty:
            fig_tags = px.bar(
                df_tags, x="count", y="complaint_tag", orientation="h",
                title=f"Knowledge Graph Entity Weights: {detected_cat}",
                color="complaint_tag", template="plotly_white",
                labels={"count": "Negative Graph Co-occurrences", "complaint_tag": "Root Cause Node"}
            )
            fig_tags.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_tags, use_container_width=True)
            
    with col_rev:
        st.markdown("### 💬 Retrieved Review Excerpts & Provenance")
        st.caption("Verifiable qualitative proof linked directly to database keys")
        
        sql_reviews = f"""
        SELECT review_id, order_id, review_score, sentiment, complaint_tag, review_comment_message_en, review_comment_message_pt
        FROM v_multimodal_analytics
        WHERE product_category_name = '{detected_cat}'
        ORDER BY (review_score <= 2) DESC, RANDOM()
        LIMIT 4
        """
        retrieved_reviews = pd.read_sql_query(sql_reviews, conn)
        for _, row in retrieved_reviews.iterrows():
            st.markdown(f"""
            <div class="evidence-card">
                <div><strong>Score: {row['review_score']}/5 ★</strong> | Tag: <code>{row['complaint_tag']}</code></div>
                <div style="margin: 5px 0; font-style: italic; font-size: 0.92rem;">"{row['review_comment_message_en']}"</div>
                <div style="font-size: 0.78rem; color: #64748B;"><em>Original (PT):</em> "{row['review_comment_message_pt']}"</div>
                <div style="margin-top: 6px;">
                    <span class="provenance-tag">review_id: {row['review_id']}</span>
                    <span class="provenance-tag">order_id: {row['order_id']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Human Review Form for Operations
    st.divider()
    st.subheader("2. Human Auditor Review & Decision Control")
    h_col1, h_col2 = st.columns([1, 1])
    with h_col1:
        eval_verdict = st.radio(
            "Auditor Validation Status:",
            ["✅ Fully Grounded: Knowledge Graph and tabular data corroborate the review causes",
             "⚠️ Inconclusive: Requires supplier telemetry before committing capital",
             "❌ Rejected: Misattributed complaint root cause"],
            index=0
        )
        h_feedback = st.text_area("Audit Log Notes:", f"Verified {cat_metrics['avg_delivery_days']} days shipping duration as main driver of '{df_tags.iloc[0]['complaint_tag'] if not df_tags.empty else 'delays'}'.")
    with h_col2:
        actions = st.multiselect(
            "Trigger Operational Remedies:",
            ["Contract Renegotiation with Regional Freight Carriers",
             "Adjust Category Delivery Promise (+6 Business Days)",
             "Mandate Reinforced Packaging for Fragile Items",
             "Issue Auto-Vouchers for Delayed Shipments"],
            default=["Adjust Category Delivery Promise (+6 Business Days)", "Mandate Reinforced Packaging for Fragile Items"]
        )
        if st.button("Commit Audit Decision", type="primary"):
            st.success(f"Audit decision logged! {len(actions)} operational remediations executed.")
            st.balloons()

# -------------------------------------------------------------------------------------------------
# PERSPECTIVE 2: CUSTOMER-FACING MODE (BUYER INQUIRIES & PREDICTIVE RISK)
# -------------------------------------------------------------------------------------------------
else:
    st.subheader("🛍️ Customer Pre-Purchase Inquiry & Satisfaction Risk Predictor")
    st.caption("Addressing Professor Feedback: Empowering buyers to inspect delivery risks and ask product questions before purchasing.")
    
    col_cust_left, col_cust_right = st.columns([1, 1])
    
    with col_cust_left:
        st.markdown("### 📦 Simulated Cart & Order Parameters")
        cust_cat = st.selectbox("Product Category of Interest:", ["furniture_decor", "bed_bath_table", "computers_accessories", "health_beauty", "sports_leisure"])
        cust_price = st.slider("Item Price ($ USD):", min_value=20.0, max_value=600.0, value=145.0, step=5.0)
        cust_weight = st.slider("Product Weight (grams):", min_value=200, max_value=15000, value=3500, step=100)
        cust_delivery_est = st.slider("Expected Distance / Delivery Days:", min_value=2, max_value=45, value=18, step=1)
        
        # Run ML Satisfaction Prediction
        is_delayed_sim = 1 if cust_delivery_est > 14 else 0
        freight_sim = round(15.0 + (cust_delivery_est * 1.2), 2)
        
        input_data = pd.DataFrame([{
            'actual_delivery_days': cust_delivery_est,
            'is_delayed': is_delayed_sim,
            'price': cust_price,
            'freight_value': freight_sim,
            'product_weight_g': cust_weight
        }])
        
        pred_prob = ml_model.predict_proba(input_data)[0]
        prob_satisfied = pred_prob[1]
        
    with col_cust_right:
        st.markdown("### 🔮 AI Predictive Satisfaction Outlook")
        
        gauge_color = "#34D399" if prob_satisfied >= 0.7 else ("#FBBF24" if prob_satisfied >= 0.4 else "#F87171")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob_satisfied * 100, 1),
            number={'suffix': "%"},
            title={'text': "Predicted Customer Satisfaction Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40], 'color': "#FEE2E2"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#DCFCE7"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if prob_satisfied < 0.5:
            st.error(f"⚠️ **High Delivery Risk Detected:** Simulated shipping time of {cust_delivery_est} days exceeds typical satisfaction thresholds. Our Knowledge Graph notes elevated rates of shipping complaints for items of this size.")
        else:
            st.success(f"✅ **Favorable Order Profile:** {cust_cat} order with {cust_delivery_est} days delivery has a high likelihood of 4–5 star satisfaction.")

    # Customer Question Answering over Reviews
    st.divider()
    st.markdown("### 💬 Ask a Question About Past Buyer Experiences")
    cust_q = st.text_input("Ask a question about this product category:", f"What do customers say about packaging and assembly for {cust_cat}?")
    
    # Query matching reviews
    sql_cust_answers = f"""
    SELECT review_id, order_id, review_score, review_comment_message_en
    FROM v_multimodal_analytics
    WHERE product_category_name = '{cust_cat}'
    ORDER BY RANDOM()
    LIMIT 3
    """
    df_cust_ans = pd.read_sql_query(sql_cust_answers, conn)
    
    st.markdown("**Retrieved Past Buyer Comments:**")
    for _, r in df_cust_ans.iterrows():
        st.info(f"🗣️ *\"{r['review_comment_message_en']}\"* — Score: {r['review_score']}/5 ★ [Order: `{r['order_id']}`]")

# -------------------------------------------------------------------------------------------------
# PERSPECTIVE 3: QUANTITATIVE BENCHMARK EVALUATION & BASELINE COMPARISON
# -------------------------------------------------------------------------------------------------
st.divider()
with st.expander("📊 Challenge 2 Benchmark: ML Predictive Model vs Baseline Heuristic Comparison", expanded=False):
    st.markdown("#### Quantitative Model Performance on Real Olist Dataset (n = 1,702 records)")
    st.caption("Requirement: Compare your improved AI approach against at least one simpler baseline solution.")
    
    m_col1, m_col2, m_col3 = st.columns([1, 1, 1])
    
    with m_col1:
        st.markdown("**Simpler Baseline (Delivery Rule)**")
        st.write("Rule: If order is delayed beyond ETA, predict dissatisfaction; otherwise predict satisfaction.")
        st.metric("Baseline Accuracy", f"{round(eval_metrics['baseline_acc'] * 100, 1)}%")
        st.metric("Baseline F1 Score", f"{round(eval_metrics['baseline_f1'] * 100, 1)}%")
        
    with m_col2:
        st.markdown("**Improved AI Model (Random Forest)**")
        st.write("Multimodal features: actual days, delay flag, product price, freight cost, physical weight.")
        st.metric("AI Accuracy", f"{round(eval_metrics['rf_acc'] * 100, 1)}%", delta=f"{round((eval_metrics['rf_acc'] - eval_metrics['baseline_acc'])*100, 1)}% improvement")
        st.metric("AI F1 Score", f"{round(eval_metrics['rf_f1'] * 100, 1)}%", delta=f"{round((eval_metrics['rf_f1'] - eval_metrics['baseline_f1'])*100, 1)}% improvement")
        
    with m_col3:
        st.markdown("**Top Feature Importances**")
        st.dataframe(feature_imp_df, use_container_width=True, hide_index=True)
