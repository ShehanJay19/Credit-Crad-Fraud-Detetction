from flask import Flask, render_template, request,jsonify
import pandas as pd
import joblib

app=Flask(__name__)

model=joblib.load('models/rf_model.pkl')

@app.route('/')
def home():
    return render_template("single_prediction.html")

@app.route('/predict', methods=['POST'])

def predict():
    data=request.get_json()
    
    df_input=pd.DataFrame([[data['Time'],data['Amount']]],columns=['Time','Amount'])
    proba=model.predict_proba(df_input)[:,1][0]
    pred=int(proba>0.5)
    return jsonify({"fraud_probability":float(proba),"is_fraud":pred})
    
@app.route('/batch', methods=["GET"])
def batch_page():
    return render_template("batch_upload.html")

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in requests.files:
        return "No file part", 400
    
    f=request.files['file']
    df=pd.read_csv(f)
    
    df_input=df[['Time','Amount']]
    proba=model.predict_proba(df_input)[:,1]
    df['fraud_probability']=proba
    df['prediction']=(proba>0.5).astype(int)
    
    return df.to_csv(index=False), 200, {'Content-Type': 'text/csv; charset=utf-8'}


if __name__ == "__main__":
    app.run(debug=True, port=5001)

    