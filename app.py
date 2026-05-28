import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Smart Text Classifier",
    page_icon="📃",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_models():
    model_files = {
        "label_encoder": "label_encoder.pkl",
        "tfidf_vectorizer": "tfidf_vectorizer.pkl",
        "logistic_regression": "model_logistic_regression.pkl",
        "naive_bayes": "model_naive_bayes.pkl",
        "random_forest": "model_random_forest.pkl",
        "svm": "model_svm.pkl",
        "xgboost": "model_xgboost.pkl",
    }
    models = {}
    for name, path in model_files.items():
        if Path(path).exists():
            models[name] = joblib.load(path)
        else:
            st.error(f"File {path} not found. Make sure it is in the same folder.")
            st.stop()
    return models

data = load_models()
category_encoder = data["label_encoder"]
vectorizer = data["tfidf_vectorizer"]

model_dict = {
    "Logistic Regression": data["logistic_regression"],
    "Naive Bayes": data["naive_bayes"],
    "Random Forest": data["random_forest"],
    "SVM": data["svm"],
    "XGBoost": data["xgboost"],
}

category_list = list(category_encoder.classes_)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Cairo', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f0f4fa 0%, #e2e8f0 100%);
    }

    .stTextArea textarea {
        border-radius: 20px;
        border: 1px solid #cbd5e1;
        font-size: 16px;
        padding: 15px;
        transition: all 0.3s ease;
        background-color: #ffffff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .stTextArea textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.2);
        outline: none;
    }

    .stButton button {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 40px;
        padding: 12px 28px;
        transition: all 0.3s ease;
        border: none;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        background: linear-gradient(90deg, #3b82f6, #1e3a8a);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .prediction-card {
        background: white;
        border-radius: 30px;
        padding: 25px 20px;
        box-shadow: 0 20px 35px -12px rgba(0,0,0,0.15);
        margin-top: 25px;
        text-align: center;
        transition: all 0.2s ease;
        border-left: 8px solid #2563eb;
    }

    .probability-bar {
        background-color: #e2e8f0;
        border-radius: 30px;
        height: 28px;
        margin: 15px 0;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .prob-fill {
        background: linear-gradient(90deg, #3b82f6, #1e3a8a);
        height: 100%;
        border-radius: 30px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 12px;
        color: white;
        font-weight: bold;
        font-size: 14px;
        transition: width 0.5s ease;
    }

    .metric-card {
        background: white;
        border-radius: 24px;
        padding: 20px 10px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        border-bottom: 4px solid #2563eb;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #1e293b;
        margin: 10px 0 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .footer {
        text-align: center;
        margin-top: 60px;
        padding: 20px;
        font-size: 13px;
        color: #475569;
        border-top: 1px solid #cbd5e1;
    }

    h1 {
        text-align: center;
        background: linear-gradient(120deg, #1e293b, #2563eb);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sidebar .sidebar-content {
        background: #ffffffcc;
        backdrop-filter: blur(5px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("### ⚙️ Settings")
    page = st.radio("📄 Navigate", ["🧠 Classifier", "📊 Evaluation"], index=0)
    st.markdown("---")
    
    selected_model_name = st.selectbox(
        "🤖 Choose Model",
        list(model_dict.keys()),
        index=0,
        help="Select the machine learning model to use for prediction."
    )
    selected_model = model_dict[selected_model_name]
    
    st.markdown("---")
    rating = st.slider("⭐ Rating (1–5)", min_value=1, max_value=5, value=3,
                       help="Product rating (only used for classification)")
    selected_category = st.selectbox("📂 Category", category_list,
                                     help="Product category (only used for classification)")
    category_encoded = category_encoder.transform([selected_category])[0]
    

if page == "🧠 Classifier":
    st.title("📃 Smart Text Classifier")
    st.markdown(
        """
        <p style="text-align: center; font-size: 1.2rem; color: #1e293b;">
        Classify reviews as <strong style="color:#2563eb;">Real (OR)</strong> or <strong style="color:#dc2626;">Fake (CG)</strong><br>
        using machine learning models.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    st.subheader("✍️ Enter Review Text")
    user_input = st.text_area(
        "",
        height=200,
        placeholder="e.g., This product exceeded my expectations! Highly recommended.",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button("Predict", use_container_width=True)
    
    if predict_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to classify.")
        else:
            input_tfidf = vectorizer.transform([user_input])
            extra_features = csr_matrix([[rating, category_encoded]])
            input_final = hstack([input_tfidf, extra_features])
            
            pred_num = selected_model.predict(input_final)[0]
            pred_label = "Real Review (OR)" if pred_num == 1 else "Fake Review (CG)"
            pred_color = "#2563eb" if pred_num == 1 else "#dc2626"
            
            max_prob = None
            prob_df = None
            if hasattr(selected_model, "predict_proba"):
                try:
                    proba = selected_model.predict_proba(input_final)[0]
                    max_prob = np.max(proba) * 100
                    prob_df = pd.DataFrame({
                        "class": ["Fake (CG)", "Real (OR)"],
                        "probability": proba * 100
                    }).sort_values("probability", ascending=False)
                except:
                    pass
            
            with st.container():
                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h3 style="margin:0 0 10px 0; color:#1e293b;">⭐ Prediction Result</h3>
                        <p style="font-size: 48px; font-weight: 800; margin: 10px 0; color:{pred_color};">{pred_label}</p>
                    """,
                    unsafe_allow_html=True,
                )
                
                if max_prob is not None and prob_df is not None:
                    st.markdown(
                        f"""
                        <div style="margin-top: 5px;">
                            <strong>Confidence:</strong> {max_prob:.2f}%
                            <div class="probability-bar">
                                <div class="prob-fill" style="width: {max_prob:.2f}%;">
                                    {max_prob:.1f}%
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    fig = px.bar(
                        prob_df,
                        x="probability",
                        y="class",
                        orientation="h",
                        title="📊 Class Probabilities",
                        labels={"probability": "Probability (%)", "class": ""},
                        color="probability",
                        color_continuous_scale="Blues",
                        text="probability"
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ℹ️ The selected model does not provide probability estimates. Only the class is shown.")
                
                st.markdown("</div>", unsafe_allow_html=True)

elif page == "📊 Evaluation":
    st.title("📊 Model Performance Evaluation")
    st.markdown(
        """
        <p style="text-align: center; font-size: 1.1rem; color: #1e293b;">
        Compare the accuracy, precision, recall, and F1‑score of all trained models.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    eval_data = {
        "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest", "XGBoost"],
        "Accuracy (%)": [89, 87, 90, 86, 84],
        "Precision (%)": [89, 87, 90, 85, 84],
        "Recall (%)": [89, 87, 90, 86, 84],
        "F1 Score (%)": [89, 87, 90, 86, 84],
    }
    df_eval = pd.DataFrame(eval_data)
    
    best_idx = df_eval["Accuracy (%)"].idxmax()
    best_model = df_eval.loc[best_idx, "Model"]
    best_acc = df_eval.loc[best_idx, "Accuracy (%)"]
    st.success(f"🏆 **Best Performing Model:** {best_model}  —  Accuracy: {best_acc}%")
    
    st.markdown("---")
    
    st.markdown(f"### 🎯 Selected Model: `{selected_model_name}`")
    model_row = df_eval[df_eval["Model"] == selected_model_name].iloc[0]
    
    cols = st.columns(4)
    metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    for col, metric in zip(cols, metrics):
        value = model_row[metric]
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{metric.replace(' (%)','')}</div>
                <div class="metric-value">{value}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    st.markdown("### 📈 Accuracy Comparison Across Models")
    fig_acc = px.bar(
        df_eval,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        text="Accuracy (%)",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Model Accuracy (%)"
    )
    fig_acc.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_acc.update_layout(
        height=450,
        yaxis_range=[75, 100],
        showlegend=False,
        xaxis_title="",
        yaxis_title="Accuracy (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 All Metrics Comparison")
    df_melted = df_eval.melt(
        id_vars="Model",
        value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"],
        var_name="Metric",
        value_name="Score"
    )
    df_melted["Metric"] = df_melted["Metric"].str.replace(" (%)", "", regex=False)
    
    fig_group = px.bar(
        df_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Performance Metrics by Model"
    )
    fig_group.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_group.update_layout(
        height=500,
        yaxis_range=[75, 100],
        xaxis_title="",
        yaxis_title="Score (%)",
        legend_title="Model",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_group, use_container_width=True)
    
    st.markdown("---")
    
    styled_df = df_eval.set_index("Model").style.highlight_max(color='#bfdbfe', axis=0).format("{:.0f}")
    st.dataframe(styled_df, use_container_width=True)

st.markdown(
    """
    <div class="footer">
        🧠 Models: Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost &nbsp;|&nbsp;
        ⚡ Real vs Fake Review Classification
    </div>
    """,
    unsafe_allow_html=True,
)