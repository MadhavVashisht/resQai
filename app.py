import gradio as gr
import pandas as pd
import xgboost as xgb
import json
import traceback
from datetime import datetime

print("--- Script starting in production mode ---")

# --- Load Model ---
model = xgb.XGBClassifier()
try:
    model.load_model("xgb_disaster_classifier.json")
    print("--- Model loaded successfully! ---")
except Exception as e:
    print(f"--- CRITICAL ERROR: Model loading failed! ---\n{e}")
    model = None

disaster_map = {0: "Flood", 1: "Storm", 2: "Earthquake", 3: "Epidemic", 4: "Landslide", 5: "Drought", 6: "Extreme temperature", 7: "Wildfire", 8: "Volcanic activity", 9: "Other"}

# --- Main Prediction Function ---
def generate_action_plan(incident_data_str: str):
    try:
        if model is None:
            raise ValueError("Model is not loaded. Check server logs.")

        incident_data = json.loads(incident_data_str)
        location = incident_data.get("position", [0.0, 0.0])
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        features = [0.5] * 271 # Placeholder feature vector
        
        X_input = pd.DataFrame([features])
        pred_class_idx = model.predict(X_input)[0]
        predicted_type = disaster_map.get(int(pred_class_idx), "Unknown")
        
        plan = (
            f"ACTION PLAN (30 MINS): A potential '{predicted_type}' event has been identified near coordinates {tuple(location)} at {time}. "
            f"Incident Title: '{incident_data.get('title', 'N/A')}'. "
            f"Priority: {incident_data.get('priority', 'N/A')}."
        )
        return {"predicted_disaster_type": predicted_type, "summary": plan}

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# --- Gradio Interface ---
iface = gr.Interface(
    fn=generate_action_plan,
    inputs=[
        gr.Textbox(label="Incident Data (as a single JSON object)", lines=15)
    ],
    outputs=gr.JSON(label="📄 Action Plan"),
    title="ResQAI - 30-Minute Disaster Response Plan"
)

# NOTE: The iface.launch() block must be removed for Railway deployment.
# Gunicorn serves the 'iface' object directly.
