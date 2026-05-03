from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("res_model.joblib")

app = FastAPI(title="AI Hiring Decision Engine")

# ---------------------------
# CORS (Frontend connect)
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
# Home Route
# ---------------------------
@app.get("/")
def home():
    return {"message": "AI Hiring API Running 🚀"}


# ---------------------------
# Analyze Route
# ---------------------------
@app.post("/analyze")
def analyze(data: InputData):

    # Convert to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Normalize text (avoid case issue)
    input_df["education_level"] = input_df["education_level"].str.lower().str.strip()
    input_df["company_type"] = input_df["company_type"].str.lower().str.strip()

    # ---------------------------
    # Prediction
    # ---------------------------
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= 0.5 else 0

    # ---------------------------
    # Reason Engine
    # ---------------------------
    reasons = []

    if data.cgpa < 6:
        reasons.append("Low CGPA")

    if data.projects < 2:
        reasons.append("Less projects")

    if data.internships == 0:
        reasons.append("No internship")

    if data.skills_score < 50:
        reasons.append("Weak technical skills")

    if data.soft_skills_score < 50:
        reasons.append("Weak communication")

    if data.experience_years < 1:
        reasons.append("Low experience")

    if data.programming_languages < 2:
        reasons.append("Low coding exposure")

    if data.resume_length_words < 300:
        reasons.append("Short resume")

    # ---------------------------
    # Risk Score
    # ---------------------------
    risk_score = 1 - proba

    if risk_score > 0.6:
        risk_level = "High Risk 🔴"
    elif risk_score > 0.3:
        risk_level = "Medium Risk 🟡"
    else:
        risk_level = "Low Risk 🟢"

    # ---------------------------
    # Final Output
    # ---------------------------
    return {
        "prediction": int(prediction),
        "result": "Hired ✅" if prediction == 1 else "Not Hired ❌",
        "confidence": float(round(proba, 3)),
        "risk_score": float(round(risk_score, 3)),
        "risk_level": risk_level,
        "reasons": reasons if reasons else ["Strong profile"]
    }





    
        
        
    
