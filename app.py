import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("car_price_pipeline.pkl")

model = load_model()

st.title("🚗 Used Car Price Predictor")
st.write(
    "Enter the car details below to estimate its selling price "
    "using the trained Random Forest regression pipeline."
)

col1, col2 = st.columns(2)

with col1:
    car_age = st.number_input(
        "Car Age (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=2_000_000,
        value=50_000,
        step=1_000
    )

    mileage = st.number_input(
        "Mileage",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=0.1
    )

    engine = st.number_input(
        "Engine Capacity (CC)",
        min_value=0.0,
        max_value=10_000.0,
        value=1200.0,
        step=10.0
    )

    max_power = st.number_input(
        "Maximum Power (bhp)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

with col2:
    seats = st.number_input(
        "Number of Seats",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=1.0
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Diesel", "Petrol", "LPG", "CNG"]
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Individual", "Dealer", "Trustmark Dealer"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    )

if st.button("Predict Selling Price", type="primary", use_container_width=True):

    car = pd.DataFrame({
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats],
        "car_age": [car_age]
    })

    try:
        prediction = model.predict(car)[0]

        st.success("Prediction completed!")
        st.metric(
            "Estimated Selling Price",
            f"₹{prediction:,.0f}"
        )

        with st.expander("Show entered car details"):
            st.dataframe(car, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error("Prediction failed. Please check that the saved pipeline matches this app.")
        st.exception(error)

st.caption(
    "Portfolio project • Used Car Price Prediction • Random Forest Regression"
)
