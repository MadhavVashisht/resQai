import gradio as gr
import pandas as pd
import xgboost as xgb
import json
import traceback
from datetime import datetime
from pydantic import BaseModel
from typing import List

# ... (The first part of your script remains the same) ...
print("--- Script starting in production mode ---")

model = xgb.XGBClassifier()
try:
    model.load_model("xgb_disaster_classifier.json")
    print("--- Model loaded successfully! ---")
except Exception as e:
    print(f"--- CRITICAL ERROR: Model loading failed! ---\n{e}")
    model = None

disaster_map = {0: "Flood", 1: "Storm", 2: "Earthquake", 3: "Epidemic", 4: "Landslide", 5: "Drought", 6: "Extreme temperature", 7: "Wildfire", 8: "Volcanic activity", 9: "Other"}

# --- MODIFIED: The core function now takes 4 inputs ---
def generate_action_plan(location_str, time, active_incidents, resources_str):
    try:
        if model is None:
            raise ValueError("Model is not loaded.")
        
        location = json.loads(location_str)
        resources = json.loads(resources_str)

        # --- MODIFIED: Features list is hardcoded here ---
        features = [0.5] * 271
        
        X_input = pd.DataFrame([features])
        pred_class_idx = model.predict(X_input)[0]
        predicted_type = disaster_map.get(int(pred_class_idx), "Unknown")
        plan = f"ACTION PLAN (30 MINS): A potential '{predicted_type}' event identified..."
        return {"predicted_disaster_type": predicted_type, "summary": plan}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# --- MODIFIED: Gradio interface now accepts 4 inputs ---
iface = gr.Interface(
    fn=generate_action_plan,
    inputs=[
        gr.Textbox(label="📍 Location (as JSON string)"),
        gr.Textbox(label="⏰ Time (ISO format)"),
        gr.Number(label="🔥 Active Incidents"),
        gr.Textbox(label="🚒 Resources (as JSON string with quoted keys)")
    ],
    outputs=gr.JSON(label="📄 Action Plan"),
    title="ResQAI - 30-Minute Disaster Response Plan"
)

# --- FastAPI Endpoints ---
app = iface.app

# --- ✅ GET ENDPOINT ---
@app.get("/run/predict")
def generate_action_plan_get(
    location_str: str,
    time: str,
    active_incidents: int,
    resources_str: str
):
    """
    Handles GET requests by calling the main function with URL parameters.
    """
    return generate_action_plan(location_str, time, active_incidents, resources_str)

# --- ✅ NEW POST ENDPOINT ---
# Define the data model for the incoming JSON body
class PredictPayload(BaseModel):
    data: List[
        str,  # location_str
        str,  # time
        int,  # active_incidents
        str   # resources_str
    ]

@app.post("/run/predict")
def generate_action_plan_post(payload: PredictPayload):
    """
    Handles POST requests by parsing the JSON body and calling the main function.
    """
    location_str, time, active_incidents, resources_str = payload.data
    return generate_action_plan(location_str, time, active_incidents, resources_str)
