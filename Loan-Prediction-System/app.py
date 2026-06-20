from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import os
import sys

try:
    from main import predict_single, load_model_safe
except ImportError:
    sys.exit(1)

app = FastAPI(title="Loan Prediction API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}

@app.on_event("startup")
def load_models():
    paths = {
        "regression": "model/model_regressor.joblib",
        "classification": "model/model_classifier.joblib"
    }
    for task, path in paths.items():
        if os.path.exists(path):
            MODELS[task] = load_model_safe(path)

class LoanApplication(BaseModel):
    task: str
    annual_inc: float
    dti: float
    revol_bal: float
    revol_util: float
    open_acc: float
    total_acc: float
    pub_rec: float
    home_ownership: str
    purpose: str
    term: str
    verification_status: str
    application_type: str
    loan_amnt: float = 0.0

@app.get("/", response_class=HTMLResponse)
def root_index():
    return """
    <html>
        <head>
            <title>Loan Prediction API</title>
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f8fafc; margin: 0; }
                .card { background: white; padding: 3rem 5rem; border-radius: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,0.05); text-align: center; border: 2px solid #e2e8f0; }
                h1 { color: #1e293b; margin-bottom: 0.5rem; }
                h2 { color: #2563eb; margin-top: 0; font-weight: 800; font-size: 2.5rem; letter-spacing: -0.025em; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Loan Prediction API is Active</h1>
                <h2>Built by Sreehari 🚀</h2>
            </div>
        </body>
    </html>
    """

@app.post("/predict")
def predict_risk(application: LoanApplication):
    task_type = application.task
    
    if task_type not in MODELS:
        raise HTTPException(status_code=503, detail=f"{task_type.capitalize()} model not found.")
    
    try:
        input_data = application.dict()
        results = predict_single(MODELS[task_type], input_data, task=task_type)
        
        return {
            "status": "success",
            "task_type": task_type,
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
