import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎨 **Custom CSS for a Beautiful UI**
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    h1 {
        text-align: center;
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
    }
    .stTextArea, .stFileUploader {
        border-radius: 10px;
        box-shadow: 0 0 10px #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 18px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# 📌 **Page Title**
st.title("🚀 Resume Ranking System")

col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

with col2:
    st.header("📝 Job Description")
    job_description = st.text_area("Enter the job description...")

# 📌 **Extract Text from PDF Resumes**
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip()

# 📌 **Rank Resumes Using TF-IDF & Cosine Similarity**
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_desc_vector], resume_vectors).flatten()

# 📌 **AI Suggestions for Resume Improvement**
def generate_resume_tips(score):
    if score > 80:
        return "🔥 Excellent match! Your resume is well-optimized."
    elif score > 60:
        return "✅ Good match! Consider adding more relevant keywords."
    else:
        return "⚡ Low match! Try improving your skills section and adding industry-specific terms."

# 📌 **Start Ranking Process**
if uploaded_files and job_description:
    st.header("📊 Resume Rankings")

    resumes = [extract_text_from_pdf(file) for file in uploaded_files]

    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    scores = rank_resumes(job_description, resumes)

    results_df = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files],
        "Match Score (%)": (scores * 100).round(2),
        "AI Suggestion": [generate_resume_tips(score * 100) for score in scores]
    }).sort_values(by="Match Score (%)", ascending=False)

    st.dataframe(results_df.style.format({"Match Score (%)": "{:.2f}"}))
    st.success(f"✅ Ranking Complete! 🎯 Top Match: **{results_df.iloc[0]['Resume']}**")
    
