from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("res_model.joblib")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class InputData(BaseModel):
    education_level: str
    cgpa: float 
    internships: int
    projects: int
    programming_languages: int
    certifications: int
    experience_years: float
    hackathons: int
    research_papers: int
    skills_score: float
    soft_skills_score: float
    resume_length_words: int
    company_type: str


@app.get("/")
def home():
    return {"message": "Hire Prediction API Running 🚀"}


@app.post("/predict")
def predict(data: InputData):

    # normalize text (IMPORTANT)
    education = data.education_level.strip().lower()
    company = data.company_type.strip().lower()

    input_df = pd.DataFrame([{
        "education_level": education,
        "cgpa": data.cgpa,
        "internships": data.internships,
        "projects": data.projects,
        "programming_languages": data.programming_languages,
        "certifications": data.certifications,
        "experience_years": data.experience_years,
        "hackathons": data.hackathons,
        "research_papers": data.research_papers,
        "skills_score": data.skills_score,
        "soft_skills_score": data.soft_skills_score,
        "resume_length_words": data.resume_length_words,
        "company_type": company
    }])

    pred = model.predict(input_df)[0]

    return {
        "prediction": int(pred),
        "result": "Hired ✅" if pred == 1 else "Not Hired ❌"
    }
