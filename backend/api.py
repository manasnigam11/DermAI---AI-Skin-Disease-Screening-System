from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import io
import os
import urllib.request
from datetime import datetime

app = FastAPI(title="DermAI – Intelligent Skin Disease Early Screening System")

# ── CORS — Allow ALL common frontend origins ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("best_model.keras")

with open("classes.json", "r") as f:
    class_indices = json.load(f)

class_names = sorted(class_indices, key=class_indices.get)
IMG_SIZE = 224

# --------------- Symptom matching profiles ---------------
# Maps each disease to its expected symptom pattern.
# The actual classes are: Eczema, Keratosis, Nevi, Normal, Psoriasis, SkinCancer, Tinea, Warts
SYMPTOM_PROFILE = {
    "Eczema":     {"itching": True,  "redness": True,  "burning": False, "allergy": True,  "spreading": False, "dry_flaky": True},
    "Psoriasis":  {"itching": True,  "redness": True,  "burning": False, "allergy": False, "spreading": True,  "dry_flaky": True},
    "Tinea":      {"itching": True,  "redness": True,  "burning": False, "allergy": False, "spreading": True,  "dry_flaky": False},
    "Warts":      {"itching": False, "redness": False, "burning": False, "allergy": False, "spreading": True,  "dry_flaky": False},
    "Keratosis":  {"itching": False, "redness": False, "burning": False, "allergy": False, "spreading": False, "dry_flaky": True},
    "Nevi":       {"itching": False, "redness": False, "burning": False, "allergy": False, "spreading": False, "dry_flaky": False},
    "SkinCancer": {"itching": False, "redness": True,  "burning": True,  "allergy": False, "spreading": True,  "dry_flaky": True},
    "Normal":     {"itching": False, "redness": False, "burning": False, "allergy": False, "spreading": False, "dry_flaky": False},
    "Ringworm":   {"itching": True,  "redness": True,  "burning": False, "allergy": False, "spreading": True,  "dry_flaky": False},
    "Acne":       {"itching": False, "redness": True,  "burning": True,  "allergy": False, "spreading": False, "dry_flaky": False},
    "Rosacea":    {"itching": False, "redness": True,  "burning": True,  "allergy": False, "spreading": False, "dry_flaky": False},
}


# --------------- In-memory history storage ---------------
# In production, use a database. For this project, we use a simple dict.
history_store: Dict[str, list] = {}

HISTORY_FILE = "scan_history.json"

def load_history_file():
    global history_store
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history_store = json.load(f)
        except:
            history_store = {}

def save_history_file():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_store, f, indent=2)
    except:
        pass

load_history_file()


# --------------- Pydantic models ---------------
class PredictionItem(BaseModel):
    disease: str
    confidence: float


class RefineRequest(BaseModel):
    predictions: List[PredictionItem]
    symptoms: Dict[str, bool]


class SymptomCheckRequest(BaseModel):
    predictions: List[PredictionItem]
    symptoms: Dict[str, bool]
    user_id: Optional[str] = None


# --------------- /predict ---------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: str = Form(default="anonymous")):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    top_indices = preds.argsort()[-3:][::-1]
    predictions = [
        {"disease": class_names[i], "confidence": round(float(preds[i]), 4)}
        for i in top_indices
    ]

    # Build symptom questions list for the frontend
    questions = list(SYMPTOM_PROFILE.get(predictions[0]["disease"], {}).keys())
    if not questions:
        questions = ["itching", "redness", "burning", "allergy", "spreading", "dry_flaky"]

    return {"predictions": predictions, "questions": questions}


# --------------- /refine ---------------
@app.post("/refine")
async def refine(body: RefineRequest):
    return _do_refine(body.predictions, body.symptoms)


# --------------- /symptom-check (alias of /refine with history save) ---------------
@app.post("/symptom-check")
async def symptom_check(body: SymptomCheckRequest):
    result = _do_refine(body.predictions, body.symptoms)

    # Save to history
    user_id = body.user_id or "anonymous"
    if user_id not in history_store:
        history_store[user_id] = []

    history_store[user_id].insert(0, {
        "timestamp": datetime.now().isoformat(),
        "predictions": [p.dict() for p in body.predictions],
        "symptoms": body.symptoms,
        "results": result["results"],
    })

    # Keep last 50 entries per user
    history_store[user_id] = history_store[user_id][:50]
    save_history_file()

    return result


def _do_refine(predictions, symptoms):
    adjusted = []

    for pred in predictions:
        disease = pred.disease if hasattr(pred, 'disease') else pred['disease']
        conf = pred.confidence if hasattr(pred, 'confidence') else pred['confidence']

        if disease in SYMPTOM_PROFILE:
            profile = SYMPTOM_PROFILE[disease]
            for symptom_key, expected_value in profile.items():
                if symptom_key in symptoms:
                    user_value = symptoms[symptom_key]
                    if user_value == expected_value:
                        conf *= 1.15   # boost for match
                    else:
                        conf *= 0.90   # penalize for mismatch

        adjusted.append({"disease": disease, "confidence": conf})

    # Normalize so confidences sum to 1.0
    total = sum(item["confidence"] for item in adjusted)
    if total > 0:
        for item in adjusted:
            item["confidence"] = item["confidence"] / total

    # Assign labels
    results = []
    for item in adjusted:
        prob = item["confidence"]
        if prob >= 0.60:
            label = "Highly Likely"
        elif prob >= 0.30:
            label = "Possible"
        else:
            label = "Maybe"

        results.append({
            "disease": item["disease"],
            "probability": round(prob, 4),
            "label": label,
        })

    # Sort descending by probability
    results.sort(key=lambda x: x["probability"], reverse=True)

    return {"results": results}


# --------------- /history ---------------
@app.get("/history")
async def get_history(user_id: str = Query(default="anonymous")):
    entries = history_store.get(user_id, [])
    return {"history": entries}


# --------------- /derma (Nearby dermatologists) ---------------
@app.get("/derma")
async def get_dermatologists(lat: float = Query(default=28.6139), lng: float = Query(default=77.2090)):
    """Returns a list of nearby dermatologists. Uses reverse geocoding to simulate results near the user."""
    location_name = "your area"
    
    # Simple reverse geocoding using Nominatim (OpenStreetMap)
    try:
        req = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}",
            headers={"User-Agent": "DermAI-App/1.0"}
        )
        res = urllib.request.urlopen(req, timeout=3)
        data = json.loads(res.read())
        address = data.get("address", {})
        
        # Priority fallback for city/locality name
        location_name = address.get("city") or address.get("town") or address.get("suburb") or address.get("county") or address.get("state") or "your area"
    except Exception as e:
        print(f"[DermAI - Geocoding Failed]: {e}")
        pass

    doctors = [
        {"name": "Dr. Priya Sharma", "specialty": "Dermatology & Cosmetology", "address": f"Apollo Hospital, {location_name}", "distance": "2.3 km", "rating": 4.8, "phone": "+91-9876543210"},
        {"name": "Dr. Rajesh Kumar", "specialty": "Clinical Dermatology", "address": f"Max Super Speciality Hospital, {location_name}", "distance": "4.1 km", "rating": 4.6, "phone": "+91-9123456789"},
        {"name": "Dr. Anita Gupta", "specialty": "Dermatology & Venereology", "address": f"Fortis Skin Clinic, {location_name}", "distance": "5.7 km", "rating": 4.7, "phone": "+91-9988776655"},
        {"name": "Dr. Vikram Singh", "specialty": "Pediatric Dermatology", "address": f"City Medical Center, {location_name}", "distance": "6.2 km", "rating": 4.9, "phone": "+91-9876501234"},
        {"name": "Dr. Meena Patel", "specialty": "Skin & Hair Specialist", "address": f"Care Hospital, {location_name}", "distance": "8.4 km", "rating": 4.5, "phone": "+91-9012345678"},
    ]
    return {"doctors": doctors, "location": {"lat": lat, "lng": lng}, "location_name": location_name}
