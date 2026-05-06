import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# --- 1. Load the MASSIVE Hybrid Dataset ---
dataset_path = 'data/gesture_dataset_large.csv'

if not os.path.exists(dataset_path):
    print(f"Error: Could not find {dataset_path}.")
    exit()

print("Loading the massive hybrid dataset... (This might take a moment)")

# --- FIX: Dynamically generate the missing headers ---
header_columns = ['label']
for i in range(21):
    header_columns.extend([f'x{i}', f'y{i}', f'z{i}'])

# Read the file and force Pandas to apply our headers
df = pd.read_csv(dataset_path, header=None, names=header_columns)

# Separate Features (X) and Labels (y)
X = df.drop('label', axis=1)
y = df['label']

# --- 2. The Exam Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train the End-Term Model ---
print(f"Training Random Forest on {len(X_train)} rows of data...")
# n_estimators=200 and n_jobs=-1 makes it powerful and fast
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)

# --- 4. Grade the AI ---
print("\n--- Grading the Model ---")
y_predict = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict)
print(f"Final AI Accuracy: {accuracy * 100:.2f}%\n")

# Prints the professional statistics table for your End-Term report
print("Detailed Classification Report:")
print(classification_report(y_test, y_predict))

# --- 5. Save the Final 'Brain' ---
model_filename = 'gesture_model_final.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\nSUCCESS: Production model saved as '{model_filename}'!")