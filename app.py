import gradio as gr
import pandas as pd
import xgboost as xgb
import json
import traceback
from datetime import datetime

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

# --- MODIFIED: Removed 'features_str' from function signature ---
def generate_action_plan(location_str, time, active_incidents, resources_str):
    try:
        if model is None:
            raise ValueError("Model is not loaded.")
        
        location = json.loads(location_str)
        resources = json.loads(resources_str)

        # --- MODIFIED: Features list is now hardcoded ---
        features = [0.5] * 271
        
        X_input = pd.DataFrame([features])
        pred_class_idx = model.predict(X_input)[0]
        predicted_type = disaster_map.get(int(pred_class_idx), "Unknown")
        plan = f"ACTION PLAN (30 MINS): A potential '{predicted_type}' event identified..."
        return {"predicted_disaster_type": predicted_type, "summary": plan}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# --- MODIFIED: Removed the 'features_str' input textbox ---
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

# --- ✅ NEW SECTION TO ADD A GET ENDPOINT ---
app = iface.app

# --- MODIFIED: Removed 'features_str' from the function signature and parameter list ---
@app.get("/run/predict")
def generate_action_plan_get(
    location_str: str,
    time: str,
    active_incidents: int,
    resources_str: str
):
    """
    This function creates a GET endpoint that accepts URL query parameters
    and then calls our original prediction function.
    """
    # Call the original function with the parameters from the URL
    return generate_action_plan(location_str, time, active_incidents, resources_str)
