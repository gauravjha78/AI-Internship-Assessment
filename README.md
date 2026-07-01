# AI-Internship-Assessment
# AI Internship Assessment — Tourism Demand Prediction & Resource Allocation 
# Link : https://tourism-ai-advisor.streamlit.app/

**Candidate:** Gaurav Avdhesh Jha
**Email:** gauravjha9878@gmail.com
**College:** Pillai College of Engineering — Electronics and Computer Science

## Overview

This project builds AI models to help tourism authorities predict demand levels and recommend resource allocation strategies, using a dataset of 15,000 tourism records across 52 columns.

## Repository Structure

```
notebooks/   → project.ipynb (full ML pipeline: EDA, preprocessing, training, evaluation)
report/      → Technical_Report.docx (detailed write-up of methodology and results)
models/      → trained model artifacts (.pkl) used by the app
app/         → Streamlit app for live predictions (Task 5)
data/        → source dataset (CSV)
```

## Task 1 — Data Exploration & Preprocessing

- Dataset checked for missing values and duplicates: none found, no imputation needed.
- Four columns dropped before modeling to prevent data leakage (`record_id`, `tourism_demand_score`, `resource_pressure_score`, `public_management_action`) — these either directly encode the target or are outcomes unavailable at prediction time.
- Seven categorical columns (`province`, `tourism_city`, `destination_type`, `season`, `weather_condition`, `tourism_event_type`, `transport_mode_priority`) encoded using Label Encoding, chosen over One-Hot Encoding because of high cardinality (up to 20 unique values) and because tree-based models don't require ordinal assumptions.
- Four engineered features added:
  - `revenue_per_tourist` = tourism_revenue_yuan / tourist_arrivals
  - `international_ratio` (international_share) = international_tourists / tourist_arrivals
  - `infra_score` (infra_composite) = average of infrastructure_availability_score, transport_service_score, accommodation_capacity_score
  - `revenue_efficiency` = tourism_revenue_yuan / operational_cost_yuan
- Stratified 80/20 train/test split on `demand_category` to preserve class balance (the minority `Very_High` class is only ~1.4% of records).

## Task 2 — Tourism Demand Prediction

Four models trained and compared on `demand_category` (Low / Moderate / High / Very_High): a most-frequent-class baseline, Decision Tree, Random Forest, and XGBoost.

**Best model: XGBoost — F1 score 0.9451** (+158% relative improvement over baseline). Selected for its sequential error-correction mechanism and strong handling of class imbalance, especially for the minority `Very_High` class.

## Task 3 — Resource Allocation Recommendation

Same four-model comparison repeated for `resource_allocation_decision` (8 classes).

**Best model: XGBoost — F1 score 0.9860** (+361% relative improvement over baseline), reflecting strong underlying patterns the model was able to learn.

## Task 4 — AI Insights

Full detailed answers are in `report/Technical_Report.docx`. Summary:

1. **Demand drivers**: `tourist_arrivals` is the strongest predictor, followed by digital signals (`online_search_trend_index`, `social_media_sentiment_score`) and `hotel_occupancy_rate` — demand is observable in advance through digital footprints.
2. **Resource allocation drivers**: `public_safety_score` and `crowd_risk_index` together account for over 42% of model decisions, making safety the dominant driver, followed by transport/infrastructure/heritage capacity scores.
3. **Revenue drivers**: `economic_impact_score` has the strongest correlation with revenue (r = 0.714); engineered features `revenue_efficiency` and `revenue_per_tourist` also rank highly, validating the feature engineering.
4. **Crowd risk vs. revenue**: correlation is effectively zero (r = 0.0032) — authorities can implement strict crowd-control measures (staggered entry, capacity limits, smart ticketing) without sacrificing revenue.
5. **Low-demand regions**: arrivals and digital engagement lag far behind high-demand regions, but promotional spend is nearly identical between the two groups — the bottleneck is marketing *effectiveness*, not budget. Recommended fix: targeted digital campaigns, modernized e-ticketing, and active online reputation management.

## Task 5 — AI Recommendation System

A Streamlit app (`app/app.py`) that takes core tourism indicators as input — Tourist Arrivals, Hotel Occupancy Rate, Tourist Satisfaction Score, Crowd Risk Index, Infrastructure Availability Score, with an optional "Advanced inputs" panel for the remaining features — and outputs:

- Predicted Demand Category
- Recommended Resource Allocation Decision
- Prediction Confidence (probability) for each

Fields not entered default to dataset-average values pulled from the saved `StandardScaler`, and the engineered features are recomputed live from current inputs so predictions stay consistent.

### Running the app

```
cd app
pip install -r requirements.txt
streamlit run app.py
```

(Requires the `models/` folder, populated with `scaler.pkl`, `feature_names.pkl`, `label_encoders.pkl`, `label_encoder_demand.pkl`, `label_encoder_resource.pkl`, `model_demand.pkl`, `model_resource.pkl`, to sit one level above `app/`.)

## Tools Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Jupyter Notebook.
