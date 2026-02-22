Resume Scanner with Skill Suggestion System -

This project is a smart Resume Scanner built using Neural Networks (Sentence-BERT) and Streamlit.
It compares a Resume with a Job Description and gives:
 Match Score (out of 100)
 Missing Skills
 Skill Suggestions to improve resume

The goal of this project is to understand how modern ATS (Applicant Tracking Systems) work using AI and NLP.

Features -

Upload Resume in PDF format
Paste Job Description
Get similarity score
Detect missing skills
Suggest important skills to add
Rank skills based on importance

How It Works -

Resume text is extracted from PDF.
Job Description is taken as input.
Both texts are converted into embeddings using Sentence-BERT (Neural Network).
Cosine similarity is calculated to generate a score out of 100.
Skills are extracted using a predefined skill database.
Missing skills are detected and ranked.
Suggestions are displayed to improve the resume.

Technologies Used -
Python
Streamlit
Sentence-Transformers
Scikit-learn
PyPDF2

Project Flow -

Resume (PDF) → Text Extraction
Job Description → Text Input
↓
Neural Network (Sentence-BERT)
↓
Cosine Similarity
↓
Match Score + Skill Suggestions

Installation -

Clone the repository:
git clone https://github.com/your-username/resume-scanner.git
cd resume-scanner
Install dependencies:
pip install -r requirements.txt
Run the project:
streamlit run app.py

Why I Built This :-

I built this project to:
Understand how AI-based resume screening works
Practice NLP and Transformer models
Improve my backend + ML integration skills
Build a real-world project for portfolio

Future Improvements -

Add experience-based scoring
Add education matching
Add database for dynamic skill updates
Deploy using FastAPI backend
Add dashboard for analytics

Author -
Rohan Gaikwad
BSc Computer Science
Interested in AI, ML, and Backend Development