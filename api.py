from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Home route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define mappings for categorical variables
        quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
        central_air_map = {"Y": 1, "N": 0}
        garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1}
        # Map `Neighborhood`, `GarageType`, and `Exterior2nd` as was used during training
        neighborhood_map = {value: idx for idx, value in enumerate([
            "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", 
            "NridgHt", "NAmes", "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr",
            "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"
        ])}
        garage_type_map = {value: idx for idx, value in enumerate([
            "Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "2Types"
        ])}
        exterior2nd_map = {value: idx for idx, value in enumerate([
            "VinylSd", "MetalSd", "Wd Shng", "HdBoard", "Plywood", "Wd Sdng", "CmentBd", "BrkFace", "Stucco",
            "AsbShng", "Brk Cmn", "ImStucc", "AsphShn", "Stone", "Other", "CBlock"
        ])}

        # Collect features from form
        features = [
            float(request.form.get('GrLivArea', 0)),
            float(request.form.get('LotFrontage', 0)),
            int(request.form.get('OverallCond', 0)),
            int(request.form.get('YearBuilt', 0)),
            int(request.form.get('YearRemodAdd', 0)),
            float(request.form.get('TotalBsmtSF', 0)),
            float(request.form.get('1stFlrSF', 0)),
            float(request.form.get('2ndFlrSF', 0)),
            float(request.form.get('GarageArea', 0)),
            float(request.form.get('LotArea', 0)),
            quality_map.get(request.form.get('KitchenQual', 'TA'), 3),
            quality_map.get(request.form.get('BsmtQual', 'TA'), 3),
            neighborhood_map.get(request.form.get('Neighborhood', 'NAmes'), 0),
            garage_type_map.get(request.form.get('GarageType', 'Attchd'), 0),
            central_air_map.get(request.form.get('CentralAir', 'N'), 0),
            quality_map.get(request.form.get('HeatingQC', 'TA'), 3),
            garage_finish_map.get(request.form.get('GarageFinish', 'Unf'), 1),
            exterior2nd_map.get(request.form.get('Exterior2nd', 'VinylSd'), 0)
        ]

        # Convert features into a NumPy array and reshape for the model
        processed_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(processed_features)
        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction[0]:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')



if __name__ == '__main__':
    app.run(debug = True)
