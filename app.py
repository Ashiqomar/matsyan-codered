# app.py

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import numpy as np
import pickle
import os
from pymongo import MongoClient
import bcrypt
from openai import AzureOpenAI
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Print working directory
print("📂 Current working directory:", os.getcwd())

# MongoDB Connection
MONGO_URI = "mongodb+srv://Rithanyaa:Rith212004@cluster0.7j1km.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    client = MongoClient(MONGO_URI)
    db = client['customer_attrition_db']
    users_collection = db['employees']
    client.admin.command('ping')
    print("✅ Connected to MongoDB Atlas successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not connect to MongoDB Atlas. Error: {e}")
    exit()

# Load Model and Dataset
model_path = "model.pkl"
csv_path = "Bank Customer Churn Prediction.csv"

if not os.path.exists(model_path) or not os.path.exists(csv_path):
    print(f"❌ ERROR: Missing required file(s). Ensure '{model_path}' and '{csv_path}' are in the folder:")
    print(f"   {os.getcwd()}")
    exit()

try:
    model = pickle.load(open(model_path, 'rb'))
    df_full = pd.read_csv(csv_path)
    print("✅ Model and dataset loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Failed to load model or dataset. Details: {e}")
    exit()

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
try:
    client = AzureOpenAI(
        api_key="999NRrpal6QAhvcvSgiopLJ144RnBl5t1sOs9e07lTHR7HsDLhwfJQQJ99BFAC77bzfXJ3w3AAABACOGDAFv",
        api_version="2023-05-15",
        azure_endpoint="https://bharathy-attrition-ai.openai.azure.com/"
    )
    print("✅ Azure OpenAI configured successfully")
except Exception as e:
    print(f"❌ ERROR: Azure OpenAI configuration failed - {str(e)}")

# Business logic
CREDIT_SCORE_LOW_RISK_THRESHOLD = 650

def apply_risk_and_suggestions(row):
    verdict, suggestions = "Low Risk", []
    if row['credit_score'] < CREDIT_SCORE_LOW_RISK_THRESHOLD:
        verdict = f"High Risk - Low Credit Score (< {CREDIT_SCORE_LOW_RISK_THRESHOLD})"
        suggestions.extend(["Offer free credit counseling.", "Consider proactive fee waivers."])
        if row['active_member'] == 0:
            suggestions.append("Flag for special re-engagement campaign.")
    elif row['predicted_churn'] == 1:
        verdict = "High Risk - Predicted to Churn by Model"
        suggestions.append("Manager should investigate the reason.")
        if row['active_member'] == 0:
            suggestions.append("Offer re-engagement bonus for inactivity.")
        if row['balance'] > 100000 and row['products_number'] == 1:
            suggestions.append("Suggest product diversification.")
    elif row['active_member'] == 0:
        verdict = "Attention - Inactive User"
        suggestions.extend(["Add to re-engagement email campaign.", "Offer bonus for next transaction."])
    else:
        verdict = "Low Risk - Stable Customer"
        suggestions.append("No immediate action required.")
    return verdict, suggestions

print("🚀 Performing initial data processing...")
features = df_full.drop(['churn', 'customer_id'], axis=1)
df_full['predicted_churn'] = model.predict(features)
verdicts_and_suggestions = df_full.apply(apply_risk_and_suggestions, axis=1)
df_full['final_verdict'], df_full['suggestions'] = zip(*verdicts_and_suggestions)
print("✅ Data processing complete.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def handle_login():
    data = request.get_json()
    user = users_collection.find_one({"_id": data.get('userId')})
    if user and bcrypt.checkpw(data.get('password').encode('utf-8'), user['password']):
        session['user_id'] = user['_id']
        session['user_name'] = user.get('name', 'Employee')
        return jsonify({'success': True, 'redirect_url': url_for('dashboard')})
    return jsonify({'success': False, 'message': 'Invalid User ID or Password'}), 401

@app.route('/signup', methods=['POST'])
def handle_signup():
    data = request.get_json()
    user_id = data.get('userId')
    if users_collection.find_one({"_id": user_id}):
        return jsonify({'success': False, 'message': 'User ID already exists.'}), 409
    hashed_password = bcrypt.hashpw(data.get('password').encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "_id": user_id,
        "password": hashed_password,
        "name": data.get('name'),
        "mobile": data.get('mobile'),
        "email": data.get('email')
    })
    return jsonify({'success': True, 'message': 'Signup successful! Please login.'})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', username=session.get('user_name', 'Employee'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/all_customers')
def get_all_customers():
    customer_list = df_full[['customer_id', 'final_verdict']].to_dict(orient='records')
    return jsonify(customer_list)

@app.route('/api/customer/<int:customer_id>')
def get_customer_details(customer_id):
    customer_data = df_full[df_full['customer_id'] == customer_id]
    if customer_data.empty:
        return jsonify({'error': 'Customer not found'}), 404
    result = customer_data.to_dict(orient='records')[0]
    for key, value in result.items():
        if isinstance(value, np.generic):
            result[key] = value.item()
    return jsonify(result)

def get_customer_context(customer_id):
    customer_data = df_full[df_full['customer_id'] == customer_id]
    if customer_data.empty:
        return "No customer data available"
    row = customer_data.iloc[0]
    context = f"""
    Customer ID: {customer_id}
    Age: {row['age']}
    Credit Score: {row['credit_score']}
    Country: {row['country']}
    Balance: ${row['balance']:,.2f}
    Products: {row['products_number']}
    Active Member: {'Yes' if row['active_member'] else 'No'}
    Churn Prediction: {'High Risk' if row['predicted_churn'] else 'Low Risk'}
    Final Verdict: {row['final_verdict']}
    """
    return context

@app.route('/api/retention_data')
def get_retention_data():
    at_risk = df_full[df_full['final_verdict'].str.contains('High Risk|Attention')].copy()
    return jsonify(at_risk.to_dict(orient='records'))

@app.route('/api/chart_data')
def chart_data():
    try:
        df_copy = df_full.copy()
        age_bins = [18, 30, 40, 50, 60, 100]
        age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        df_copy['age_group'] = pd.cut(df_copy['age'], bins=age_bins, labels=age_labels, right=False)
        age_churn_data = df_copy.groupby('age_group')['churn'].value_counts(normalize=True).unstack().fillna(0)
        country_churn_data = df_copy.groupby('country')['churn'].value_counts(normalize=True).unstack().fillna(0)
        avg_scores = df_copy.groupby('churn')['credit_score'].mean()
        chart_payload = {
            "age_group_churn": {"labels": age_churn_data.index.tolist(), "data": (age_churn_data[1] * 100).tolist()},
            "country_churn": {"labels": country_churn_data.index.tolist(), "data": (country_churn_data[1] * 100).tolist()},
            "credit_score_avg": {"labels": ["Did Not Churn", "Churned"], "data": [avg_scores.get(0, 0), avg_scores.get(1, 0)]}
        }
        return jsonify(chart_payload)
    except Exception as e:
        print(f"Error in chart_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chatbot/send', methods=['POST'])
def handle_chat_message():
    try:
        data = request.json
        customer_id = data.get('customer_id')
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        customer_data = df_full[df_full['customer_id'] == int(customer_id)]
        if customer_data.empty:
            return jsonify({"error": "Customer not found"}), 404

        row = customer_data.iloc[0]
        context = f"""
        Customer ID: {customer_id}
        Age: {row['age']}
        Credit Score: {row['credit_score']}
        Country: {row['country']}
        Balance: ${row['balance']:,.2f}
        Products: {row['products_number']}
        Active Member: {'Yes' if row['active_member'] else 'No'}
        Churn Prediction: {'High Risk' if row['predicted_churn'] else 'Low Risk'}
        Final Verdict: {row['final_verdict']}
        """

        messages = [
            {
                "role": "system",
                "content": f"""You are a banking customer retention assistant. Help the user with retention strategies.

                Customer Context:
                {context}

                Guidelines:
                - Be professional but concise
                - Focus on actionable retention strategies
                - Never share sensitive data
                - If unsure, ask for clarification"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                timeout=10
            )
            return jsonify({
                "success": True,
                "response": response.choices[0].message.content
            })
        except Exception as api_error:
            print(f"Azure OpenAI API error: {str(api_error)}")
            return jsonify({
                "success": False,
                "response": "I'm having temporary technical difficulties. Please try again in a moment."
            })

    except Exception as e:
        print(f"Error in chatbot endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "response": "Our AI service is currently unavailable. Please contact support if this persists."
        }), 500

@app.route('/api/test_azure')
def test_azure():
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Say 'hello world'"}],
            max_tokens=10
        )
        return jsonify({"success": True, "response": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
