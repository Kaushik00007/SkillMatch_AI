import streamlit as st
import pdfplumber
import docx
import re
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

st.set_page_config(page_title="Resume Rank", layout="wide")
st.sidebar.title(" About")
st.sidebar.info("Resume Rank is an resume screening & ranking system that helps recruiters match candidates with job descriptions using AI-powered analysis.")
st.sidebar.title(" FAQ")
st.sidebar.info("**How does this work?**\nWe use AI to analyze resumes and match them to job descriptions based on skills and experience.")

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            st.error(f"Error extracting text from {file.name}: {str(e)}")
    elif file.name.endswith(".docx"):
        try:
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            st.error(f"Error extracting text from {file.name}: {str(e)}")
    return text.strip()

def extract_skills(text):
    skills_list = ["python", "java", "c++", "machine learning", "deep learning", "nlp", "sql", "aws", "git", "tensorflow", "pytorch", "excel", "javascript", "html", "css", "react", "angular", "node.js"]
    return list(set([skill for skill in skills_list if skill in text.lower()]))

def calculate_similarity(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc] + resumes)
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten() * 100  

st.title("Resume Rank - AI Resume Ranking System")
st.markdown("Effortlessly match top talent with the right job!")

job_description = st.text_area("Job Description", "", height=150)
uploaded_files = st.file_uploader("📂 Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not uploaded_files or not job_description:
        st.warning("Please upload resumes and enter a job description!")
    else:
        resumes_text = [extract_text(file) for file in uploaded_files]
        valid_files = [uploaded_files[i] for i, text in enumerate(resumes_text) if text.strip()]
        resumes_text = [text for text in resumes_text if text.strip()]

        if not resumes_text:
            st.error("No valid text extracted from resumes!")
        else:
            scores = calculate_similarity(job_description, resumes_text)
            skills_match = [extract_skills(text) for text in resumes_text]
            
            ranked_results = sorted(zip(valid_files, scores, skills_match), key=lambda x: x[1], reverse=True)
            results_df = pd.DataFrame(
                [(file.name, score, ", ".join(skills)) for file, score, skills in ranked_results],
                columns=["Resume", "Match Score (%)", "Extracted Skills"]
            )
            
            st.subheader("🏆 Top Matching Resumes")
            for file, score, skills in ranked_results:
                with st.expander(f" {file.name} - {score:.2f}% Match"):
                    st.write(f"**Match Score:** {score:.2f}%")
                    st.write(f"**Extracted Skills:** {', '.join(skills)}")
                    st.download_button("Download Resume", file, file.name)
            
            st.subheader("Resume Match Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(y=[file.name for file in valid_files], x=scores, palette="Blues_r", ax=ax)
            ax.set_xlabel("Match Score (%)")
            ax.set_title("Resume Match Score Comparison")
            st.pyplot(fig)
            
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button("Download Ranked Results", csv_buffer.getvalue(), "ranked_resumes.csv", "text/csv")
