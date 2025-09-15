import gradio as gr
import pandas as pd
import xgboost as xgb
import json

print("--- 1. Script starting ---")

# --- Load the trained XGBoost model ---
# This method is reliable and avoids version mismatch errors.
model = xgb.XGBClassifier()
try:
    print("--- 2. Attempting to load model from xgb_disaster_classifier.json ---")
    model.load_model("xgb_disaster_classifier.json")
    print("--- 3. Model loaded successfully! ---")
except Exception as e:
    print(f"--- X. CRITICAL ERROR: Model loading failed! ---")
    print(e)
    model = None

# --- Mapping from numerical prediction to disaster name ---
# This should match the labels from your training data.
disaster_map = {
    0: "Flood",
    1: "Storm",
    2: "Earthquake",
    3: "Epidemic",
    4: "Landslide",
    5: "Drought",
    6: "Extreme temperature",
    7: "Wildfire",
    8: "Volcanic activity",
    9: "Other"
}

# --- Main prediction function ---
def generate_action_plan(location, time, active_incidents, features, resources):
    """
    Predicts disaster type from input features and generates a 30-minute action plan.
    """
    print("--- 4. generate_action_plan function CALLED ---")
    
    if model is None:
        print("--- 5a. Model is None, returning error. ---")
        return {"error": "Model file not found or failed to load. Check the logs."}

    print("--- 5b. Model is loaded, proceeding with prediction. ---")
    
    # The model expects a specific number of features.
    # Update '9' if your model expects a different number.
    if len(features) != 9:
        return {"error": f"Feature vector must have 9 elements, but received {len(features)}."}

    # Convert the feature list into a pandas DataFrame for the model.
    X_input = pd.DataFrame([features])
    
    # Predict the numerical class index.
    pred_class_idx = model.predict(X_input)[0]
    
    # Convert the index to a human-readable name.
    predicted_type = disaster_map.get(int(pred_class_idx), "Unknown")

    # Generate the summary text.
    plan = (
        f"ACTION PLAN (30 MINS): A potential '{predicted_type}' event has been identified at coordinates {tuple(location)} at {time}. "
        f"Current Status: {int(active_incidents)} active incidents reported. "
        f"Immediate Action: Deploy available resources ({resources}). "
        f"Primary Objectives: Prioritize evacuation, public alerts, and first aid deployment."
    )
    
    print("--- 6. Prediction successful, returning plan. ---")
    return {"predicted_disaster_type": predicted_type, "summary": plan}

# --- Create the Gradio Web Interface ---
iface = gr.Interface(
    fn=generate_action_plan,
    inputs=[
        gr.JSON(label="📍 Location (e.g., [19.07, 72.87])"),
        gr.Textbox(label="⏰ Time (ISO format, e.g., 2025-09-16T08:00:00Z)"),
        gr.Number(label="🔥 Active Incidents (e.g., 3)"),
        gr.JSON(label="📊 Feature Vector (list of 9 numbers)"),
        gr.JSON(label="🚒 Resources (JSON object)")
    ],
    outputs=gr.JSON(label="📄 Action Plan"),
    title="ResQAI - 30-Minute Disaster Response Plan",
    description="This AI bot uses a feature vector to predict the disaster type and generates an immediate, 30-minute action plan."
)

# --- Launch the application ---
if __name__ == "__main__":
    iface.launch()