# 📄 AI-Powered Resume Screening & Ranking System

##  Introduction
The **AI-Powered Resume Screening & Ranking System** is an intelligent tool designed to streamline the recruitment process. It enables recruiters to efficiently analyze and rank multiple resumes against a given job description using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **cosine similarity**. This ensures an objective and accurate matching of candidates based on relevant skills and experience.

## 🎯 Features
☑️ **Multi-Resume Upload** – Supports batch processing of resumes in PDF and DOCX formats.  
☑️ **TF-IDF & Cosine Similarity** – Ensures accurate ranking based on keyword relevance.  
☑️ **Resume Parsing & Analysis** – Extracts key details such as skills, experience, and certifications.  
☑️ **Graphical Match Comparison** – Visualizes resume ranking with a bar chart.  
☑️ **Bias-Free Hiring (Resume Anonymization)** – Option to hide personal details for unbiased screening.  
☑️ **User-Friendly UI** – Built using **Streamlit** for an intuitive and interactive experience.  
☑️ **Downloadable Reports** – Provides ranked results in CSV format for further analysis.  

## 🛠 Tech Stack
- **Python** – Core language
- **Streamlit** – Frontend UI
- **TfidfVectorizer** – Resume ranking
- **Cosine Similarity** – Similarity scoring
- **pdfplumber & python-docx** – Resume parsing
- **Matplotlib & Seaborn** – Data visualization

## 📌 Installation
Clone the repository:
```bash
git clone https://github.com/Kaushik00007/AI-powered-Resume-Screening-and-Ranking-System
cd AI-powered-Resume-Screening-and-Ranking-System
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run ResumeRank.py
```

## 🖥️ Usage
1. Enter or paste the **job description** in the text area.
2. Upload **multiple resumes** in PDF/DOCX format.
3. Click the **"Match Resumes"** button to analyze and rank them.
4. View **top matching resumes**, extracted skills, and match scores.
5. Download the **ranked results** in CSV format.

## 🔮 Future Enhancements
- **NER-based Resume Parsing** using advanced NLP models.
- **Experience & Certification Weightage** for better ranking.
- **Integration with ATS Systems** for seamless recruitment.
- **Support for More File Formats** (TXT, JSON, etc.).

## 🙌 Contributions
Contributions are welcome! Follow these steps:
```
Fork the repository
Create a new branch 
Commit your changes
Open a pull request
```
## 📧 Contact
For any queries, reach out via:

- 📧 Email: kaushi00007@gmail.com  
- 🔗 LinkedIn: https://www.linkedin.com/in/kaushik-k-dev
- 🌍 GitHub: https://github.com/Kaushik00007/Kaushik00007

## Built with using Python, Machine Learning, and Streamlit. 
