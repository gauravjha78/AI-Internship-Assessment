"""
Task 5 - AI Recommendation System
Tourism Demand Prediction & Resource Allocation Recommendation

Run locally with:
    streamlit run app.py

Expects the following files in a sibling "models" folder
(rename the downloaded .pkl files to exactly these names):
    models/scaler.pkl
    models/feature_names.pkl
    models/label_encoders.pkl              (dict of LabelEncoders for categorical columns)
    models/label_encoder_demand.pkl
    models/label_encoder_resource.pkl
    models/model_demand.pkl
    models/model_resource.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    le_demand = joblib.load(os.path.join(MODELS_DIR, "label_encoder_demand.pkl"))
    le_resource = joblib.load(os.path.join(MODELS_DIR, "label_encoder_resource.pkl"))
    model_demand = joblib.load(os.path.join(MODELS_DIR, "model_demand.pkl"))
    model_resource = joblib.load(os.path.join(MODELS_DIR, "model_resource.pkl"))
    return scaler, feature_names, label_encoders, le_demand, le_resource, model_demand, model_resource


scaler, feature_names, label_encoders, le_demand, le_resource, model_demand, model_resource = load_artifacts()

# Dataset-mean defaults for every numeric feature the scaler was trained on,
# used for any field the user doesn't explicitly set.
DEFAULTS = dict(zip(feature_names, scaler.mean_))

CATEGORICAL_COLS = list(label_encoders.keys())
CATEGORICAL_DEFAULTS = {
    "province": "Beijing",
    "tourism_city": "Beijing",
    "destination_type": "Urban_Commercial_Tourism_Hub",
    "season": "Summer",
    "weather_condition": "Clear",
    "tourism_event_type": "No_Major_Event",
    "transport_mode_priority": "Private_Car",
}

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Tourism Demand & Resource Allocation AI", layout="centered")
st.title("🧭 Tourism Demand & Resource Allocation Predictor")
st.caption("AI Internship Assessment — Task 5: AI Recommendation System")

st.markdown("Enter the core tourism indicators below. Everything else uses dataset-average values unless you expand **Advanced inputs**.")

col1, col2 = st.columns(2)
with col1:
    tourist_arrivals = st.number_input("Tourist Arrivals", min_value=0, value=60000, step=1000)
    hotel_occupancy_rate = st.slider("Hotel Occupancy Rate (%)", 0.0, 100.0, 67.0)
    tourist_satisfaction_score = st.slider("Tourist Satisfaction Score (0-100)", 0.0, 100.0, 65.0)
with col2:
    crowd_risk_index = st.slider("Crowd Risk Index (0-100)", 0.0, 100.0, 51.0)
    infrastructure_availability_score = st.slider("Infrastructure Availability Score (0-100)", 0.0, 100.0, 62.0)

with st.expander("Advanced inputs (optional — improves accuracy)"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        province = st.selectbox("Province", label_encoders["province"].classes_,
                                 index=list(label_encoders["province"].classes_).index(CATEGORICAL_DEFAULTS["province"]))
        tourism_city = st.selectbox("Tourism City", label_encoders["tourism_city"].classes_,
                                     index=list(label_encoders["tourism_city"].classes_).index(CATEGORICAL_DEFAULTS["tourism_city"]))
        destination_type = st.selectbox("Destination Type", label_encoders["destination_type"].classes_,
                                         index=list(label_encoders["destination_type"].classes_).index(CATEGORICAL_DEFAULTS["destination_type"]))
        season = st.selectbox("Season", label_encoders["season"].classes_,
                               index=list(label_encoders["season"].classes_).index(CATEGORICAL_DEFAULTS["season"]))
        weather_condition = st.selectbox("Weather Condition", label_encoders["weather_condition"].classes_,
                                          index=list(label_encoders["weather_condition"].classes_).index(CATEGORICAL_DEFAULTS["weather_condition"]))
    with adv_col2:
        tourism_event_type = st.selectbox("Tourism Event Type", label_encoders["tourism_event_type"].classes_,
                                           index=list(label_encoders["tourism_event_type"].classes_).index(CATEGORICAL_DEFAULTS["tourism_event_type"]))
        transport_mode_priority = st.selectbox("Transport Mode Priority", label_encoders["transport_mode_priority"].classes_,
                                                index=list(label_encoders["transport_mode_priority"].classes_).index(CATEGORICAL_DEFAULTS["transport_mode_priority"]))
        international_tourists = st.number_input("International Tourists", min_value=0, value=int(DEFAULTS["international_tourists"]))
        public_safety_score = st.slider("Public Safety Score (0-100)", 0.0, 100.0, float(DEFAULTS["public_safety_score"]))
        transport_service_score = st.slider("Transport Service Score (0-100)", 0.0, 100.0, float(DEFAULTS["transport_service_score"]))

    accommodation_capacity_score = st.slider("Accommodation Capacity Score (0-100)", 0.0, 100.0, float(DEFAULTS["accommodation_capacity_score"]))
    tourism_revenue_yuan = st.number_input("Tourism Revenue (Yuan)", min_value=0.0, value=float(DEFAULTS["tourism_revenue_yuan"]))
    operational_cost_yuan = st.number_input("Operational Cost (Yuan)", min_value=1.0, value=float(DEFAULTS["operational_cost_yuan"]))
    online_search_trend_index = st.slider("Online Search Trend Index (0-100)", 0.0, 100.0, float(DEFAULTS["online_search_trend_index"]))
    social_media_sentiment_score = st.slider("Social Media Sentiment Score (0-1)", 0.0, 1.0, float(DEFAULTS["social_media_sentiment_score"]))
    e_ticket_booking_rate = st.slider("E-Ticket Booking Rate (0-100)", 0.0, 100.0, float(DEFAULTS["e_ticket_booking_rate"]))

predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Build feature vector + predict
# ----------------------------------------------------------------------
if predict_btn:
    row = dict(DEFAULTS)  # start from dataset means for every column

    # overwrite with user-provided primary inputs
    row["tourist_arrivals"] = tourist_arrivals
    row["hotel_occupancy_rate"] = hotel_occupancy_rate
    row["tourist_satisfaction_score"] = tourist_satisfaction_score
    row["crowd_risk_index"] = crowd_risk_index
    row["infrastructure_availability_score"] = infrastructure_availability_score

    # overwrite with advanced inputs
    row["international_tourists"] = international_tourists
    row["public_safety_score"] = public_safety_score
    row["transport_service_score"] = transport_service_score
    row["accommodation_capacity_score"] = accommodation_capacity_score
    row["tourism_revenue_yuan"] = tourism_revenue_yuan
    row["operational_cost_yuan"] = operational_cost_yuan
    row["online_search_trend_index"] = online_search_trend_index
    row["social_media_sentiment_score"] = social_media_sentiment_score
    row["e_ticket_booking_rate"] = e_ticket_booking_rate

    # encode categoricals
    cat_values = {
        "province": province, "tourism_city": tourism_city, "destination_type": destination_type,
        "season": season, "weather_condition": weather_condition,
        "tourism_event_type": tourism_event_type, "transport_mode_priority": transport_mode_priority,
    }
    for col, val in cat_values.items():
        row[col] = label_encoders[col].transform([val])[0]

    # recompute engineered features from current inputs
    row["revenue_per_tourist"] = row["tourism_revenue_yuan"] / max(row["tourist_arrivals"], 1)
    row["international_share"] = row["international_tourists"] / max(row["tourist_arrivals"], 1)
    row["infra_composite"] = np.mean([
        row["infrastructure_availability_score"],
        row["transport_service_score"],
        row["accommodation_capacity_score"],
    ])
    row["revenue_efficiency"] = row["tourism_revenue_yuan"] / max(row["operational_cost_yuan"], 1)
    row["digital_engagement"] = np.mean([
        row["online_search_trend_index"],
        row["social_media_sentiment_score"] * 100,
        row["e_ticket_booking_rate"],
    ])

    # assemble in correct column order
    X = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)
    X_scaled = scaler.transform(X)

    demand_probs = model_demand.predict_proba(X_scaled)[0]
    demand_pred_idx = int(np.argmax(demand_probs))
    demand_label = le_demand.inverse_transform([demand_pred_idx])[0]
    demand_confidence = demand_probs[demand_pred_idx]

    resource_probs = model_resource.predict_proba(X_scaled)[0]
    resource_pred_idx = int(np.argmax(resource_probs))
    resource_label = le_resource.inverse_transform([resource_pred_idx])[0]
    resource_confidence = resource_probs[resource_pred_idx]

    st.divider()
    st.subheader("📊 Prediction Results")

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Predicted Demand Category", demand_label)
        st.progress(float(demand_confidence), text=f"Confidence: {demand_confidence*100:.1f}%")
    with r2:
        st.metric("Recommended Resource Allocation", resource_label)
        st.progress(float(resource_confidence), text=f"Confidence: {resource_confidence*100:.1f}%")

    with st.expander("Show full class probabilities"):
        st.write("**Demand Category**")
        st.dataframe(pd.DataFrame({"class": le_demand.classes_, "probability": demand_probs}).sort_values("probability", ascending=False), hide_index=True)
        st.write("**Resource Allocation Decision**")
        st.dataframe(pd.DataFrame({"class": le_resource.classes_, "probability": resource_probs}).sort_values("probability", ascending=False), hide_index=True)
