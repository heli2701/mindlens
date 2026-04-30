"""
MindLens — Hugging Face Space
FastAPI serves the HTML frontend at / and the prediction API at /predict
"""

import os
import json
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MAX_LEN = 128
DEVICE = torch.device("cpu")   # HF free Spaces = CPU only
HF_REPO = os.getenv("HF_MODEL_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# ── Download model from HF Hub if not present ──────────────────
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    weights_path = os.path.join(MODEL_DIR, "mindlens_roberta.pt")

    if os.path.exists(weights_path):
        print("✅ Model already present.")
        return

    if not HF_REPO:
        raise RuntimeError(
            "HF_MODEL_REPO secret is missing.\n"
            "Set it in Space Settings > Variables and secrets.\n"
            "Example: heli18/mindlens-weights"
        )

    print(f"⏳ Downloading model from {HF_REPO} ...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        local_dir=MODEL_DIR,
        token=HF_TOKEN,
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print("✅ Model downloaded.")


download_model()

# ── Load label map ──────────────────────────────────────────────
label_map_path = os.path.join(MODEL_DIR, "label_map.json")
if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"label_map.json not found at: {label_map_path}")

with open(label_map_path, "r", encoding="utf-8") as f:
    label_cfg = json.load(f)

if "id2label" not in label_cfg:
    raise KeyError("label_map.json must contain 'id2label'")

if "num_classes" not in label_cfg:
    raise KeyError("label_map.json must contain 'num_classes'")

CLASS_NAMES = [label_cfg["id2label"][str(i)] for i in range(label_cfg["num_classes"])]
NUM_CLASSES = label_cfg["num_classes"]

# Safer default
MODEL_BASE = label_cfg.get("model_base", "roberta-base")

print(f"✅ Classes: {CLASS_NAMES}")
print(f"✅ Num classes: {NUM_CLASSES}")
print(f"✅ Base model: {MODEL_BASE}")

# ── Model ───────────────────────────────────────────────────────
class MindLensModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_BASE)
        hidden = self.roberta.config.hidden_size

        # +3 because your model expects aux_features of size 3
        self.norm = nn.LayerNorm(hidden + 3)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden + 3, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask, aux_features):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_out, aux_features], dim=1)
        x = self.norm(combined)
        x = self.drop1(x)
        x = F.gelu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)

# ── Load model ──────────────────────────────────────────────────
weights_path = os.path.join(MODEL_DIR, "mindlens_roberta.pt")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at: {weights_path}")

print("⏳ Loading model weights...")
model = MindLensModel(num_classes=NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(
    weights_path,
    map_location=DEVICE,
    weights_only=False,
)

# unwrap if saved with {"state_dict": ...}
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# remove DataParallel prefix if present
clean_checkpoint = {}
for k, v in checkpoint.items():
    clean_checkpoint[k.replace("module.", "")] = v

missing, unexpected = model.load_state_dict(clean_checkpoint, strict=False)

print("✅ Weights loaded.")
if missing:
    print("⚠️ Missing keys:", missing)
if unexpected:
    print("⚠️ Unexpected keys:", unexpected)

model.eval()

# ── Load tokenizer ──────────────────────────────────────────────
print("⏳ Loading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
    tokenizer_loaded_from = f"{MODEL_BASE} (fast)"
    print(f"✅ Fast tokenizer loaded from base model: {MODEL_BASE}")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=False)
    tokenizer_loaded_from = f"{MODEL_BASE} (slow)"
    print(f"⚠️ Fallback: slow tokenizer loaded from base model: {MODEL_BASE}")

print(f"✅ Model ready on {DEVICE}")

# ── Inference ───────────────────────────────────────────────────
def run_inference(text: str) -> dict:
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    aux = torch.zeros(1, 3).to(DEVICE)

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], aux)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_id = int(probs.argmax())

    return {
        "prediction": CLASS_NAMES[pred_id],
        "confidence": round(float(probs[pred_id]), 4),
        "probabilities": {
            c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)
        },
    }

# ── SHAP ────────────────────────────────────────────────────────
class _SHAPPipe:
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        enc = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        aux = torch.zeros(len(texts), 3).to(DEVICE)

        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"], aux)
            return torch.softmax(logits, dim=-1).cpu().numpy()

_shap_pipe = _SHAPPipe()
_shap_masker = shap.maskers.Text(tokenizer)
_shap_explainer = shap.Explainer(
    _shap_pipe,
    _shap_masker,
    output_names=CLASS_NAMES,
)

def run_shap(text: str) -> list:
    shap_vals = _shap_explainer([text[:300]])
    probs = run_inference(text)["probabilities"]
    pred_id = int(np.array([probs[c] for c in CLASS_NAMES]).argmax())

    tokens = shap_vals.data[0]
    values = shap_vals.values[0, :, pred_id]

    scored = [
        {"token": t, "importance": round(float(v), 4)}
        for t, v in zip(tokens, values)
    ]
    scored.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return scored[:10]

# ── FastAPI ─────────────────────────────────────────────────────
app = FastAPI(title="MindLens")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

    @validator("text")
    def validate_text(cls, v):
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Text must be at least 10 characters.")
        if len(v) > 5000:
            raise ValueError("Text must be under 5,000 characters.")
        return v

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    top_tokens: list = []

# ── Routes ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if not os.path.exists("index.html"):
        return HTMLResponse(
            content="<h2>MindLens API is running, but index.html was not found.</h2>",
            status_code=200,
        )

    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()

    return HTMLResponse(content=html)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "tokenizer_loaded_from": tokenizer_loaded_from,
        "model_base": MODEL_BASE,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return PredictResponse(**run_inference(req.text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/explain", response_model=PredictResponse)
def predict_explain(req: PredictRequest):
    try:
        result = run_inference(req.text)
        return PredictResponse(**result, top_tokens=run_shap(req.text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
def get_classes():
    return {"classes": {str(i): n for i, n in enumerate(CLASS_NAMES)}}