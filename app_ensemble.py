import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ======================================================
# 🧠 Define the Average Ensemble class (required for loading)
# ======================================================
class AverageEnsemble:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        return (pred1 + pred2) / 2

# ======================================================
# 🎨 PAGE STYLE FUNCTION
# ======================================================
def set_page_style():
    try:
        with open("dayimg.png", "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        
        st.markdown(f"""
            <style>
                [data-testid="stAppViewContainer"] {{
                    background-image: url("data:image/png;base64,{img_data}");
                    background-size: cover;
                    background-position: center;
                }}
                [data-testid="stHeader"], [data-testid="stToolbar"] {{
                    background-color: transparent;
                }}
                .main {{
                    background-color: rgba(0, 0, 0, 0.55);
                    border-radius: 10px;
                    padding: 1rem;
                }}
                h1, h2, h3, h4, h5, h6, p, label, span {{
                    color: white !important;
                }}
                .stButton > button {{
                    background-color: #2e7d32 !important;
                    color: white !important;
                    border-radius: 5px !important;
                    padding: 0.5rem 1rem !important;
                    border: none !important;
                }}
                div[data-baseweb="select"] > div,
                div[data-testid="stNumberInput"] input,
                div[data-testid="stSelectbox"] div {{
                    background-color: rgba(255, 255, 255, 0.1) !important;
                    color: white !important;
                    border: 1px solid rgba(255, 255, 255, 0.2) !important;
                    border-radius: 4px !important;
                }}
                div[data-testid="stInfoAlert"], div[data-testid="stSuccessAlert"] {{
                    background-color: rgba(46, 125, 50, 0.2) !important;
                    color: white !important;
                }}
            </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Error loading background image: {str(e)}")

# ======================================================
# ⚙️ LOAD TRAINED ENSEMBLE MODEL
# ======================================================
model = joblib.load("final_apartment_price_model.pkl")

st.set_page_config(page_title="🏙️ Colombo Apartment Price Predictor", layout="centered")
set_page_style()

# ======================================================
# 🏠 APP TITLE
# ======================================================
st.title("🏠 Colombo Apartment Price Prediction App")
st.write("""
Predict apartment prices in **Colombo District** using the trained **Average Ensemble Model**  
(Combining Random Forest and XGBoost for improved accuracy).
""")

# ======================================================
# 🧾 USER INPUTS
# ======================================================
st.subheader("Enter Apartment Details")

colombo_code = st.selectbox("📍 Select Colombo Code (1–15)", list(range(1, 16)))
bedrooms = st.number_input("🛏 Number of Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("🛁 Number of Bathrooms", min_value=1, max_value=10, value=2)
floor_area = st.number_input("📐 Total Floor Area (sq.ft)", min_value=200, max_value=5000, value=1000)
is_premium = st.radio("🌆 Is it a Premium Area? (Colombo 1, 2, 3, 4, 7)", ["Yes", "No"])

# ======================================================
# 📍 REGION CODE MAPPING
# ======================================================
north_colombo = [11, 13, 14, 15]
south_colombo = [4, 5, 6, 7]

if colombo_code in north_colombo:
    region_code = 2
    region_name = "North Colombo"
elif colombo_code in south_colombo:
    region_code = 3
    region_name = "South Colombo"
else:
    region_code = 1
    region_name = "Central Colombo"

st.info(f"📌 Automatically selected **{region_name}** (Region Code: {region_code})")

# Convert premium area to binary
is_premium_area = 1 if is_premium == "Yes" else 0

# ======================================================
# 🧩 FEATURE ENGINEERING (same as training)
# ======================================================
floor_area_log = np.log1p(floor_area)
bedroom_per_1000sqft = (bedrooms / floor_area) * 1000
bathroom_per_bedroom = bathrooms / bedrooms

# Prepare the input DataFrame
input_data = pd.DataFrame([{
    'Colombo_Code': colombo_code,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Floor_Area_Log': floor_area_log,
    'Bedroom_per_1000sqft': bedroom_per_1000sqft,
    'Bathroom_per_Bedroom': bathroom_per_bedroom,
    'is_premium_area': is_premium_area,
    'region_code': region_code
}])

# ======================================================
# 🔮 Prediction Section (Final — Correct Scale Applied)
# ======================================================
if st.button("Predict Apartment Price"):
    # Ensure input matches training features
    input_data = pd.DataFrame([{
        'Colombo_Code': float(colombo_code),
        'Bedrooms': float(bedrooms),
        'Bathrooms': float(bathrooms),
        'Floor_Area_Log': float(np.log1p(floor_area)),
        'Bedroom_per_1000sqft': float((bedrooms / floor_area) * 1000),
        'Bathroom_per_Bedroom': float(bathrooms / bedrooms),
        'is_premium_area': float(1 if is_premium == "Yes" else 0),
        'region_code': float(region_code)
    }])

    # Predict (log scale)
    log_price_pred = model.predict(input_data)[0]
    #st.write("🧩 Debug Info — Model Raw Log Prediction:", round(log_price_pred, 3))

    # Convert log to actual price 
    predicted_price = np.expm1(log_price_pred)  
    # Sanity check
    if predicted_price < 1_000_000:
        st.error("⚠️ The predicted price seems too low. Please check inputs.")
    else:
        price_million = predicted_price / 1_000_000
        lower_bound = price_million * 0.9
        upper_bound = price_million * 1.1

        st.subheader("💰 Predicted Apartment Price")
        st.success(f" **Estimated Price:** Rs. {price_million:,.2f} million")
        st.info(f" Expected Range: Rs. {lower_bound:,.2f}M – Rs. {upper_bound:,.2f}M")

        # Context summary
        st.markdown("---")
        st.markdown(
            f"📍 **Location:** Colombo {colombo_code} ({region_name})\n\n"
            f"🏢 **Property Details:** {bedrooms} Bedrooms, {bathrooms} Bathrooms, {floor_area} sq.ft\n\n"
            f"🌆 **Premium Area:** {'Yes' if is_premium == 'Yes' else 'No'}"
        )

        # Price indicator bar
        st.markdown("### 📈 Price Level Indicator")
        normalized_value = min(1.0, price_million / 100)
        st.progress(normalized_value)


# ======================================================
# 📊 FOOTER
# ======================================================
st.markdown("---")
st.caption("Developed by G.A.A.S Ganegoda | BSc (Hons) Data Science | NSBM Green University | 2025")

