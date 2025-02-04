import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from model_utils import train_model, evaluate_model, predict_spam
from data_utils import preprocess_text, load_data, create_visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Spam Email Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("📧 Spam Email Classification")
    st.markdown("""
    This application demonstrates spam email classification using machine learning.
    You can explore the dataset, view model performance, and test the classifier with your own text.
    """)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Explorer", "Model Performance", "Predict"])

    if page == "Dataset Explorer":
        st.header("Dataset Exploration")
        
        # Load and display data
        data = load_data()
        st.write("Sample of the dataset:", data.head())
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Emails", len(data))
            st.metric("Spam Emails", len(data[data['label'] == 1]))
        with col2:
            st.metric("Ham Emails", len(data[data['label'] == 0]))
            spam_ratio = (len(data[data['label'] == 1]) / len(data)) * 100
            st.metric("Spam Ratio", f"{spam_ratio:.2f}%")

        # Show visualizations
        st.subheader("Visualizations")
        figs = create_visualizations(data)
        for fig in figs:
            st.pyplot(fig)

    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        
        # Train model and get metrics
        metrics, conf_matrix, roc_curve = evaluate_model()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['precision']:.3f}")
        col3.metric("Recall", f"{metrics['recall']:.3f}")
        col4.metric("F1 Score", f"{metrics['f1']:.3f}")
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        st.pyplot(conf_matrix)
        
        # Display ROC curve
        st.subheader("ROC Curve")
        st.pyplot(roc_curve)

    else:  # Predict page
        st.header("Test the Spam Classifier")
        
        # Text input
        user_text = st.text_area("Enter email text to classify:", height=200)
        
        if st.button("Classify Email"):
            if user_text:
                # Make prediction
                prediction, probability = predict_spam(user_text)
                
                # Display result with custom styling
                result_color = "green" if not prediction else "red"
                result_text = "HAM (Not Spam)" if not prediction else "SPAM"
                st.markdown(f"""
                    <div style='padding: 20px; background-color: {result_color}20; border-radius: 5px;'>
                        <h3 style='color: {result_color}; margin: 0;'>Classification: {result_text}</h3>
                        <p>Confidence: {probability:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to classify.")

if __name__ == "__main__":
    main()
