from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

model_path = os.path.join('models', 'best_regression_model.pkl')
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    if isinstance(data, dict):
        data = [data]
    
    df = pd.DataFrame(data)
    
    if 'school' in df.columns:
        df = df.drop(columns=['school'])
    
    preprocessor = model.named_steps['preprocessor']
    df_transformed = preprocessor.transform(df)
    
    prediction = model.named_steps['regressor'].predict(df_transformed)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)