
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List

from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("res_model.joblib")

app = FastAPI(title="AI Hiring Decision Engine PRO")

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Input Schema
# ---------------------------
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

# ---------------------------
# Home
# ---------------------------
@app.get("/")
def home():
    return {"message": "AI Hiring Engine PRO 🚀"}

# ---------------------------
# Scoring Function
# ---------------------------
def calculate_score(data):
    score = 0

    score += min(data.cgpa * 5, 35)                  # CGPA weight
    score += min(data.projects * 5, 15)              # Projects
    score += min(data.skills_score * 0.3, 15)        # Skills
    score += min(data.soft_skills_score * 0.2, 10)   # Soft skills
    score += min(data.internships * 5, 10)           # Internships
    score += min(data.experience_years * 5, 10)      # Experience
    score += min(data.programming_languages * 2, 5)  # Coding diversity

    return round(score, 2)

# ---------------------------
# Single Analyze
# ---------------------------
@app.post("/analyze")
def analyze(data: InputData):

    input_df = pd.DataFrame([data.dict()])

    input_df["education_level"] = input_df["education_level"].str.lower().str.strip()
    input_df["company_type"] = input_df["company_type"].str.lower().str.strip()

    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= 0.5 else 0

    score = calculate_score(data)

    strengths = []
    improvements = []

    if data.cgpa >= 7:
        strengths.append("Good CGPA")
    else:
        improvements.append("Improve academic performance")

    if data.projects >= 3:
        strengths.append("Strong project experience")
    else:
        improvements.append("Add more projects")

    if data.skills_score >= 60:
        strengths.append("Strong technical skills")
    else:
        improvements.append("Improve technical skills")

    if data.internships >= 1:
        strengths.append("Has internship experience")
    else:
        improvements.append("Gain internship experience")

    risk_score = 1 - proba

    if risk_score > 0.6:
        risk_level = "High Risk 🔴"
    elif risk_score > 0.3:
        risk_level = "Medium Risk 🟡"
    else:
        risk_level = "Low Risk 🟢"

    return {
        "prediction": int(prediction),
        "result": "Hired ✅" if prediction == 1 else "Not Hired ❌",
        "confidence": round(float(proba), 3),
        "score": score,
        "risk_level": risk_level,
        "strengths": strengths,
        "improvements": improvements
    }

# ---------------------------
# Batch Ranking API 🔥
# ---------------------------
@app.post("/rank")
def rank_candidates(candidates: List[InputData]):

    results = []

    for data in candidates:
        input_df = pd.DataFrame([data.dict()])

        input_df["education_level"] = input_df["education_level"].str.lower().str.strip()
        input_df["company_type"] = input_df["company_type"].str.lower().str.strip()

        proba = model.predict_proba(input_df)[0][1]
        score = calculate_score(data)

        results.append({
            "candidate": data.dict(),
            "score": score,
            "confidence": round(float(proba), 3),
            "result": "Hired" if proba >= 0.5 else "Not Hired"
        })

    # Sort by score (descending)
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    # Add rank
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "total_candidates": len(ranked),
        "ranking": ranked
    }




    
        
        
    
