import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- 1. Load the Custom Dataset ---
print("Loading custom dataset...")
try:
    df = pd.read_csv('C:\\Users\\alok8\\OneDrive\\Documents\\Gesture project\\custom_dataset.csv')
except FileNotFoundError:
    print("Error: 'custom_dataset.csv' not found. Run the data collector script first!")
    exit()

# --- 2. Separate Coordinates (X) from Labels (y) ---
# 'y' is the letter (A, B, C), 'X' is the 63 mathematical coordinates
X = df.drop('label', axis=1)
y = df['label']

# --- 3. Split into Training and Testing Data ---
# We keep 20% of the data hidden from the AI to test it later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Initialize and Train the Random Forest ---
print("Training the Random Forest AI (This might take a few seconds)...")
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# --- 5. Test the AI's Accuracy ---
print("\nEvaluating Model Accuracy...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 50)
print(f"✅ Final AI Accuracy: {accuracy * 100:.2f}%")
print("-" * 50)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Save the Model for the Web App ---
print("\nSaving the model as 'gesture_model_final.pkl'...")
with open('gesture_model_final.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Done! You are ready to launch your Web App.")