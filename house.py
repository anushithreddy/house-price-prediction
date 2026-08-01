import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset/Housing.csv")

# Convert text to numbers
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

for column in text_columns:
    data[column] = encoder.fit_transform(data[column])

# Input and Output
X = data.drop("price", axis=1)
y = data["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Predict new house price
print("\n===== House Price Prediction =====")

area = int(input("Enter Area (sq ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))
stories = int(input("Enter Number of Stories: "))

mainroad = int(input("Main Road? (1=Yes, 0=No): "))
guestroom = int(input("Guest Room? (1=Yes, 0=No): "))
basement = int(input("Basement? (1=Yes, 0=No): "))
hotwaterheating = int(input("Hot Water Heating? (1=Yes, 0=No): "))
airconditioning = int(input("Air Conditioning? (1=Yes, 0=No): "))

parking = int(input("Parking Spaces: "))
prefarea = int(input("Preferred Area? (1=Yes, 0=No): "))

print("\nFurnishing Status")
print("0 = Furnished")
print("1 = Semi-Furnished")
print("2 = Unfurnished")

furnishingstatus = int(input("Enter Choice: "))

sample_house = pd.DataFrame({
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

predicted_price = model.predict(sample_house)
print("\n🏠 Predicted House Price: ₹", round(predicted_price[0], 2))