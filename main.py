from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 🔥 CORS import
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("res_model.joblib")

app = FastAPI()

# 🔥 CORS setup (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production में specific domain देना
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
    programing_languages: int
    certifications: int
    experience_years: float
    hackathons: int
    research_papers: int
    skills_score: float
    soft_skills_score: float
    resume_length_words: int
    company_type; str

@app.get("/")
def home():
    return {"message": "Hiire Prediction API Running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    import pandas as pd

    input_df = pd.DataFrame([{
        "education_level": data,education_level,
        "cgpa": data.cgpa,
        "internships": data.internships,
        "projects": data.projects,
        "programming_languages": data.programming_languages,
        "certifications": data.certifications,
        "experience_years": data.experience_years,
        "hackathons": data.hackathons,
        "research_papers": data.research_papers,
        "skills_score": data.skills_score ,
        "soft_skills_score": data,soft_skills_scor,
        "resume_length_words"; data,resume_length_words,
        "company_type"; data,company_type

    }])

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction),
        "result": "Hired ✅" if prediction == 1 else "Nothired ❌"
    }
