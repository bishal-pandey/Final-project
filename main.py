from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import torch
import io
from preprocessing import *
from prediction import batch_predict

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextData(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("code.html", {"request": request})

@app.get("/analyzer", response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
def analyze(data: TextData):
    return run_sentiment_pipeline([data.text])  # wrap string in list

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = await file.read()

    if filename.endswith(".txt"):
        text = content.decode("utf-8").strip().splitlines()
    elif filename.endswith(".csv"):
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        text = df.astype(str).fillna("").values.flatten().tolist()
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(content))
        text = df.astype(str).fillna("").values.flatten().tolist()
    else:
        return {"error": "Unsupported file format"}
    print(run_sentiment_pipeline(text))
    return run_sentiment_pipeline(text)

def run_sentiment_pipeline(text):
    predict, sen_score = batch_predict(text)
    
    if isinstance(text, str):
        prob = [round(sen_score[0][i].item(), 2) for i in range(len(sen_score[0]))]
    else:
        probs, _ = torch.max(sen_score, dim=1)
        prob = [round(prob.item(), 2) for prob in probs]

    return {
        "sentiment_class": predict.tolist(),
        "sentiment_score": prob,
        "label": text
    }
