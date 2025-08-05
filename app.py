from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
import os
from xhtml2pdf import pisa
from io import BytesIO

# Set up paths
base_path = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_path, 'model')
app = Flask(__name__)

# Load model, scaler, and column structure
model = pickle.load(open(os.path.join(model_path, 'churn_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))
model_columns = pickle.load(open(os.path.join(model_path, 'model_columns.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_dict = request.form.to_dict()
    df = pd.DataFrame([input_dict])
    print("Input received:", input_dict)
    print("DataFrame after conversion:\n", df)

    required_fields = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    missing = [field for field in required_fields if field not in df.columns]
    if missing:
        return f"Error: Missing required fields: {missing}. Please make sure all form fields are filled."


    # Convert data types
    try:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        df['tenure'] = df['tenure'].astype(int)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['TotalCharges'] = df['TotalCharges'].astype(float)
    except Exception as e:
        return f"Data type conversion error: {str(e)}"

    # One-hot encode and align columns
    df_encoded = pd.get_dummies(df)
    print("Model columns:\n", model_columns)
    print("Form columns after encoding:\n", df_encoded.columns.tolist())
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale
    scaled = scaler.transform(df_encoded)

    # Predict
    prediction = model.predict(scaled)
    prob = model.predict_proba(scaled)[:, 1]

    result = "Customer is likely to churn." if prediction[0] == 1 else "Customer is likely to stay."
    
    # Render result.html
    return render_template('result.html', prediction=result, churn_prob=round(prob[0]*100, 2), data=input_dict)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    user_input = request.form.to_dict()
    result = request.form['result']
    churn_prob = request.form['churn_prob']
    
    # Render HTML template for PDF
    html = render_template('pdf_template.html', data=user_input, result=result, churn_prob=churn_prob)

    # Generate PDF
    result_pdf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=result_pdf)
    result_pdf.seek(0)

    return send_file(result_pdf, mimetype='application/pdf',
                     download_name='prediction_result.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
