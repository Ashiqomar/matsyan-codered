# --- All your existing imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
# --- New imports for saving the file and checking its existence ---
import pickle
import os

warnings.filterwarnings('ignore')

# --- Configuration (No changes here) ---
DATASET_PATH = r"C:\Users\Ashiq\OneDrive\Desktop\CUSTOMER ATTRITION_Final\CUSTOMER ATTRITION_Final\CUSTOMER ATTRITION\Bank Customer Churn Prediction.csv"
CREDIT_SCORE_RISK_THRESHOLD = 750 

print("--- Customer Attrition Model Training & Prediction Script ---")

# === PART 1: LOAD AND PREPROCESS DATA (No changes here) ===
print("\n[STEP 1] Loading and preprocessing data...")

# --- Adding an explicit check to make the error clearer ---
if not os.path.exists(DATASET_PATH):
    print(f"❌ FATAL ERROR: The input file was not found at '{DATASET_PATH}'")
    print("Please make sure the file path and name are correct before running again.")
    exit()

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Successfully loaded dataset from '{DATASET_PATH}'")
    print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"❌ ERROR while loading CSV: {e}")
    exit()

df_original = df.copy()
df = df.drop(['customer_id'], axis=1)
X = df.drop('churn', axis=1)
y = df['churn']
categorical_features = ['country', 'gender']
numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# === PART 2: TRAIN THE RANDOM FOREST MODEL (No changes here) ===
print("\n[STEP 2] Training the Random Forest model...")

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# === PART 3, 4, 5 (Analysis Parts - No changes, they are good for checking your work) ===
print("\n[STEP 3] Evaluating model performance on unseen test data...")
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n[STEP 4] Applying model and business rules to the full dataset...")
# ... (all your existing code for analysis remains the same) ...
all_features = df_original.drop(['churn', 'customer_id'], axis=1)
df_original['predicted_churn'] = model.predict(all_features)
def apply_risk_verdict(row):
    if row['predicted_churn'] == 1:
        return "High Risk - Predicted to Churn by Model"
    elif row['credit_score'] > CREDIT_SCORE_RISK_THRESHOLD:
        return f"Attention - High Value Customer (Credit > {CREDIT_SCORE_RISK_THRESHOLD})"
    else:
        return "Low Risk - Stable Customer"
df_original['final_verdict'] = df_original.apply(apply_risk_verdict, axis=1)
print("✅ Final verdicts generated for all customers.")
print("\n[STEP 5] Displaying results...")
print("\n--- Sample of Final Results (First 15 Customers) ---")
print(df_original[['customer_id', 'credit_score', 'churn', 'predicted_churn', 'final_verdict']].head(15).to_string())


# === PART 6: SAVE THE TRAINED MODEL (THIS IS THE NEW, CRUCIAL PART) ===
print("\n[STEP 6] Saving the trained model to 'model.pkl'...")

try:
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    # Final confirmation check
    if os.path.exists('model.pkl'):
        print("✅ SUCCESS! 'model.pkl' has been created in your project folder.")
        print("You can now run 'python app.py'")
    else:
        print("❌ CRITICAL FAILURE: The script finished, but 'model.pkl' was not created. This might be a permissions issue.")

except Exception as e:
    print(f"❌ ERROR while saving the model file: {e}")
    print("This could be due to folder permissions. Try running your command prompt as an Administrator.")

print("\n--- Script Finished ---")