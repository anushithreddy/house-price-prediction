import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Housing.csv")

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoder = LabelEncoder()

text_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

for col in text_columns:
    data[col] = encoder.fit_transform(data[col])

# -----------------------------
# Features & Target
# -----------------------------
X = data.drop("price", axis=1)
y = data["price"]

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

accuracy = model.score(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")

st.write("""
This AI application predicts the estimated house price based on the house features using the **Linear Regression** Machine Learning algorithm.
""")

# Sidebar
st.sidebar.title("📊 Model Information")
st.sidebar.write("**Algorithm:** Linear Regression")
st.sidebar.write(f"**Accuracy:** {accuracy:.2%}")
st.sidebar.write("**Dataset:** Housing.csv")
st.sidebar.write("**Language:** Python")

# Dataset
if st.checkbox("Show Dataset"):
    st.dataframe(data)

st.subheader("Enter House Details")

area = st.number_input("Area (sq ft)", min_value=500, value=7420)
bedrooms = st.number_input("Bedrooms", min_value=1, value=4)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
stories = st.number_input("Stories", min_value=1, value=3)
parking = st.number_input("Parking Spaces", min_value=0, value=2)

mainroad = st.selectbox("Main Road", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

# -----------------------------
# Convert Inputs
# -----------------------------
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

mapping = {
    "Furnished": 0,
    "Semi-Furnished": 1,
    "Unfurnished": 2
}

furnishingstatus = mapping[furnishingstatus]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    sample = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    prediction = model.predict(sample)

    st.balloons()

    st.success("💰 Estimated House Price")

    st.metric(
        label="Predicted Price",
        value=f"₹ {prediction[0]:,.0f}"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Anushith | House Price Prediction using Machine Learning")
