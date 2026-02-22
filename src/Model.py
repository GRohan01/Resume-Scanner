import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Skill Database (can expand)
# -----------------------------
SKILLS_DB = [
    "Python", "Machine Learning", "Deep Learning", "FastAPI", "Flask",
    "SQL", "MySQL", "PostgreSQL", "Docker", "AWS", "Git", "GitHub",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "Data Analysis", "REST API", "Linux"
]

# -----------------------------
# Helper Functions
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_skills(text, skills_db):
    text = text.lower()
    extracted = set()
    for skill in skills_db:
        if skill.lower() in text:
            extracted.add(skill)
    return extracted


def rank_missing_skills(missing_skills, jd_text, model):
    jd_vec = model.encode([jd_text])
    ranked = []

    for skill in missing_skills:
        skill_vec = model.encode([skill])
        relevance = cosine_similarity(skill_vec, jd_vec)[0][0]
        ranked.append((skill, relevance))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Resume Scanner with Skill Suggestions (Neural Network)")
st.write("Upload your resume and paste the Job Description to get:")
st.write("✔ Match Score (out of 100)")
st.write("✔ Skill improvement suggestions")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")
job_desc = st.text_area("Paste Job Description here")
submit_button = st.button("Analyze Resume")

# -----------------------------
# Main Logic
# -----------------------------
if submit_button and uploaded_file and job_desc:

    # Load Neural Network model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract resume text
    resume_text = extract_text_from_pdf(uploaded_file)

    # Encode texts
    resume_vec = model.encode([resume_text])
    jd_vec = model.encode([job_desc])

    # Calculate score
    score = cosine_similarity(resume_vec, jd_vec)[0][0] * 100

    st.subheader("📊 Match Score")
    st.success(f"{score:.2f} / 100")

    # -----------------------------
    # Skill Extraction
    # -----------------------------
    resume_skills = extract_skills(resume_text, SKILLS_DB)
    jd_skills = extract_skills(job_desc, SKILLS_DB)

    missing_skills = jd_skills - resume_skills

    # -----------------------------
    # Suggestions
    # -----------------------------
    st.subheader("🧠 Skill Gap Analysis")

    if missing_skills:
        ranked_skills = rank_missing_skills(missing_skills, job_desc, model)

        st.warning("To improve your resume score, consider adding these skills:")
        for skill, relevance in ranked_skills[:5]:
            st.write(f"🔹 **{skill}** (importance: {relevance*100:.1f}%)")
    else:
        st.success("✅ Your resume already covers the required skills!")
