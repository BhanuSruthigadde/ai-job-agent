import streamlit as st
import pdfplumber
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

# =========================
# MODEL (CACHED → NO LAG)
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =========================
# SESSION INIT
# =========================
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

# =========================
# PDF EXTRACTOR
# =========================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# =========================
# RESUME EMBEDDING CACHE
# =========================
@st.cache_data
def get_resume_embedding(resume):
    return model.encode(resume, convert_to_tensor=True)

# =========================
# AI MATCH SCORE (FAST + ACCURATE)
# =========================
def calculate_match(resume, job_desc):
    emb1 = get_resume_embedding(resume)
    emb2 = model.encode(job_desc, convert_to_tensor=True)

    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score * 100, 2)

# =========================
# REAL JOBS FROM ADZUNA
# =========================
@st.cache_data(ttl=3600)
def get_jobs(query):
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "what": query,
        "results_per_page": 10,
        "content-type": "application/json"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return []

    data = response.json()

    jobs = []

    for item in data.get("results", []):
        jobs.append({
            "role": item.get("title"),
            "company": item.get("company", {}).get("display_name"),
            "location": item.get("location", {}).get("display_name"),
            "description": item.get("description", "")
        })

    return jobs

# =========================
# RESUME TAILOR
# =========================
def tailor_resume(resume, job):
    return f"""
Professional Summary:
Experienced Python developer with strong backend and problem-solving skills aligned to {job['role']} role at {job['company']}.

Key Skills:
Python, SQL, Backend Development

Objective:
To contribute effectively to {job['company']} as a {job['role']}.

Note:
Resume tailored based on job requirements.
"""

# =========================
# EMAIL GENERATOR
# =========================
def generate_email(name, job):
    return f"""
Subject: Application for {job['role']} at {job['company']}

Dear {job['company']} Team,

I am excited to apply for the {job['role']} role at your organization.

I have strong experience in Python, SQL, and backend development, and I believe my skills align well with your requirements.

I am particularly interested in contributing to your team at {job['company']}.

Thank you for your time and consideration.

Best regards,  
{name}
"""

# =========================
# UI
# =========================
st.title("💼 AI Job Agent (LinkedIn + AI Style)")

query = st.text_input("Search Jobs", "python developer")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file:

    if st.session_state.resume_text is None:
        st.session_state.resume_text = extract_text_from_pdf(uploaded_file)

    st.success("Resume uploaded successfully!")

    st.subheader("📊 Top Job Matches")

    jobs = get_jobs(query)

    if not jobs:
        st.warning("No jobs found or API issue.")
    else:

        for i, job in enumerate(jobs):

            score = calculate_match(st.session_state.resume_text, job["description"])

            st.write(f"### Match Score: {score}")
            st.write(f"**Role:** {job['role']}")
            st.write(f"**Company:** {job['company']}")
            st.write(f"**Location:** {job['location']}")

            if st.button(f"Generate for {job['role']}", key=f"{i}-{job['role']}"):

                tailored = tailor_resume(st.session_state.resume_text, job)
                email = generate_email("Bhanusruthi", job)

                st.subheader("📄 Tailored Resume")
                st.text(tailored)

                st.subheader("📧 Generated Email")
                st.text(email)

                st.download_button(
                    "Download Resume",
                    tailored,
                    file_name="tailored_resume.txt"
                )

                st.download_button(
                    "Download Email",
                    email,
                    file_name="application_email.txt"
                )

else:
    st.info("Upload a PDF resume to start matching jobs.")