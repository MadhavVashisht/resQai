import gradio as gr
import pandas as pd
import xgboost as xgb
import json
import traceback

print("--- Script starting in production mode ---")

# --- Load the trained XGBoost model ---
model = xgb.XGBClassifier()
try:
    model.load_model("xgb_disaster_classifier.json")
    print("--- Model loaded successfully! ---")
except Exception as e:
    print(f"--- CRITICAL ERROR: Model loading failed! ---\n{e}")
    model = None

# ... (disaster_map and generate_action_plan function remain the same) ...
disaster_map = {0: "Flood", 1: "Storm", 2: "Earthquake", 3: "Epidemic", 4: "Landslide", 5: "Drought", 6: "Extreme temperature", 7: "Wildfire", 8: "Volcanic activity", 9: "Other"}

def generate_action_plan(location_str, time, active_incidents, features_str, resources_str):
    try:
        if model is None:
            raise ValueError("Model is not loaded.")
        location = json.loads(location_str)
        features = json.loads(features_str)
        resources = json.loads(resources_str)
        if len(features) != 271:
            raise ValueError(f"Feature vector must have 271 elements, but received {len(features)}.")
        X_input = pd.DataFrame([features])
        pred_class_idx = model.predict(X_input)[0]
        predicted_type = disaster_map.get(int(pred_class_idx), "Unknown")
        plan = f"ACTION PLAN (30 MINS): A potential '{predicted_type}' event identified..."
        return {"predicted_disaster_type": predicted_type, "summary": plan}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# --- Create the Gradio Web Interface ---
iface = gr.Interface(
    fn=generate_action_plan,
    inputs=[
        gr.Textbox(label="📍 Location (as JSON string)"),
        gr.Textbox(label="⏰ Time (ISO format)"),
        gr.Number(label="🔥 Active Incidents"),
        gr.Textbox(label="📊 Feature Vector (as JSON string of 271 numbers)"),
        gr.Textbox(label="🚒 Resources (as JSON string with quoted keys)")
    ],
    outputs=gr.JSON(label="📄 Action Plan"),
    title="ResQAI - 30-Minute Disaster Response Plan"
)

# NOTE: The iface.launch() block has been removed for Render deployment.
# Gunicorn will serve the 'iface' object directly.
