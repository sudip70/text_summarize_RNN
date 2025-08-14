import streamlit as st
import pandas as pd
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import json
import logging
import warnings
warnings.filterwarnings("ignore")

#Ensuring set_page_config is the first Streamlit command
st.set_page_config(page_title="Text Summarization App", layout="wide", initial_sidebar_state="expanded")

#Configuring logging
logging.basicConfig(level=logging.INFO, filename='summarization_combined.log')
logger = logging.getLogger(__name__)

#Downloading NLTK data if needed
nltk.download('punkt_tab')

#Loading the saved vectorizer, models, and results and caching for faster access
@st.cache_resource
def load_components():
    try:
        #Loading models and vectorizer
        vectorizer = load('tfidf_vectorizer.pkl')
        clf_dt = load('decision_tree_classifier_model.pkl')
        reg_lr = load('linear_regression_model.pkl')
        clf_lr = load('logistic_regression_model.pkl')
        reg_dt = load('decision_tree_regressor_model.pkl')
        
        #Loading LSTM model with custom objects to handle 'mse' metric
        custom_objects = {'mse': MeanSquaredError()}
        model_lstm_reg = load_model('lstm_regressor_model.h5', custom_objects=custom_objects)
        
        #Loading evaluation results from JSON file which was saved during the training
        with open('summarization_combined_results.json', 'r') as f:
            results = json.load(f)
        return vectorizer, clf_lr, clf_dt, reg_lr, reg_dt, model_lstm_reg, results
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        logger.error(f"Error loading components: {str(e)}")
        return None, None, None, None, None, None, None

vectorizer, clf_dt, reg_lr, clf_lr, reg_dt, model_lstm_reg, results = load_components()

#Cleaning text
def clean_text(text):
    text = re.sub(r'[^\w\s.]', '', text)
    return text.strip().lower()

#Generating summary for the article using the selected model
def generate_summary(article, model, vectorizer, k=3, is_classifier=True, is_lstm=False):
    try:
        sentences = nltk.sent_tokenize(clean_text(article))
        if not sentences:
            return "No sentences found."
        
        tfidf = vectorizer.transform(sentences)
        lengths = [len(s.split()) for s in sentences]
        positions = [i / len(sentences) for i in range(len(sentences))]
        features = np.hstack([tfidf.toarray(), np.array([positions, lengths]).T])
        
        #If LSTM model, expand dimensions for compatibility
        if is_lstm:
            features = np.expand_dims(features, axis=1)
        
        #For classifiers, we predict the class and select sentences accordingly
        if is_classifier:
            preds = model.predict(features)
            selected = [sentences[i] for i in range(len(sentences)) if preds[i] == 1]

        #For regressors, we predict scores and select top k sentences
        else:
            scores = model.predict(features).flatten()
            selected_indices = np.argsort(scores)[-k:]
            selected = [sentences[i] for i in selected_indices]
        
        return ' '.join(selected) if selected else "No sentences selected."
    #Exception handling with logging
    except Exception as e:
        logger.error(f"Error in generate_summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

#Function to extract and format metrics for display
def get_model_metrics(model_name, results):
    metrics = {}
    #Evaluation metrics for classification models
    if model_name in results.get('classification_metrics', {}):
        class_metrics = results['classification_metrics'][model_name]
        metrics['F1-Score'] = round(class_metrics.get('f1_score', 0), 4)
        metrics['Precision'] = round(class_metrics.get('precision', 0), 4)
        metrics['Recall'] = round(class_metrics.get('recall', 0), 4)
        metrics['Accuracy'] = round(class_metrics.get('accuracy', 0), 4)
    
    #Evaluation metrics for Regression metrics
    if model_name in results.get('regression_metrics', {}):
        reg_metrics = results['regression_metrics'][model_name]
        metrics['MSE'] = round(reg_metrics.get('mse', 0), 4)
        metrics['R²'] = round(reg_metrics.get('r2', 0), 4)
    
    #ROUGE scores for the selected models
    for rouge in results.get('rouge_scores', []):
        if rouge['model'] == model_name:
            metrics['ROUGE-1'] = round(rouge.get('rouge1', 0), 4)
            metrics['ROUGE-2'] = round(rouge.get('rouge2', 0), 4)
            metrics['ROUGE-L'] = round(rouge.get('rougeL', 0), 4)
    
    return metrics

#Custom CSS for styling the webpage
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 6px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
        font-size: 16px;
        padding: 10px;
    }
    .stSelectbox div {
        border-radius: 8px;
        font-size: 16px;
    }
    .stSlider div {
        font-size: 16px;
    }
    .summary-box {
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .metrics-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 4px solid #007bff;
    }
    h1 {
        color: #1a3c6d;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    h2, h3 {
        color: #343a40;
    }
    .stCaption {
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

#Sidebar with setting to select the model
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model",
        [  "LSTM Regressor", "Linear Regression", "Logistic Regression", "Decision Tree Regressor", "Decision Tree Classifier"],
        help="Choose a model to generate the summary."
    )
    k_sentences = st.slider(
        "Number of Sentences in Summary",
        min_value=1,
        max_value=10,
        value=3,
        help="Select the number of sentences for the summary."
    )
    st.markdown("---")
    st.caption("Text summarization app powered by scikit-learn and TensorFlow.")
    

#Main content area on the UI
st.title("📝 Text Summarization App")
st.markdown("Paste an article below, select a model, and generate a summary. View the model's performance metrics after summarization.")

#Text area for article input
article_input = st.text_area(
    "Input Article",
    height=300,
    placeholder="Paste your article here...",
    help="Enter the text you want to summarize (minimum 50 characters, maximum 5000 characters).",
    max_chars=5000
)

#Function for generate summary button
if st.button("Generate Summary"):
    if article_input.strip():
        #Setting minimum limit for the article input
        if len(article_input.strip()) < 50:
            st.error("Input article must be at least 50 characters long.")
        else:
            with st.spinner("Generating summary..."):
                #Generating summary based on selected model
                if model_choice == "Logistic Regression":
                    summary = generate_summary(article_input, clf_lr, vectorizer, k=k_sentences, is_classifier=True)
                elif model_choice == "Decision Tree Classifier":
                    summary = generate_summary(article_input, clf_dt, vectorizer, k=k_sentences, is_classifier=True)
                elif model_choice == "Linear Regression":
                    summary = generate_summary(article_input, reg_lr, vectorizer, k=k_sentences, is_classifier=False)
                elif model_choice == "Decision Tree Regressor":
                    summary = generate_summary(article_input, reg_dt, vectorizer, k=k_sentences, is_classifier=False)
                elif model_choice == "LSTM Regressor":
                    summary = generate_summary(article_input, model_lstm_reg, vectorizer, k=k_sentences, is_classifier=False, is_lstm=True)
                
                #Displaing summary
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.subheader("Generated Summary")
                st.write(summary)
                st.markdown('</div>', unsafe_allow_html=True)
                
                #Display evaluation metrics
                if results:
                    metrics = get_model_metrics(model_choice, results)
                    st.markdown('<div class="metrics-box">', unsafe_allow_html=True)
                    st.subheader(f"{model_choice} Evaluation Metrics")
                    if metrics:
                        col1, col2 = st.columns(2)
                        for metric, value in metrics.items():
                            with col1 if len(metrics) // 2 > list(metrics.keys()).index(metric) else col2:
                                st.metric(label=metric, value=value)
                    else:
                        st.warning("No evaluation metrics available for this model.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                #Error handling with else
                else:
                    st.warning("Evaluation metrics not available. Please ensure summarization_combined_results.json exists.")
    else:
        st.warning("Please enter an article to summarize.")

#Footer for the application
st.markdown("---")

st.caption("Built with Streamlit | Models trained on CNN News Data | Metrics loaded from summarization_combined_results.json")




