from flask import Flask, request, jsonify
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomClass

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        json_data = request.get_json()
        
        # Create CustomClass instance with JSON data
        data = CustomClass(
            gender=json_data.get("gender"),
            SeniorCitizen=int(json_data.get("SeniorCitizen")),
            Partner=json_data.get("Partner"),
            Dependents=json_data.get("Dependents"),
            tenure=int(json_data.get("tenure")),
            PhoneService=json_data.get("PhoneService"),
            MultipleLines=json_data.get("MultipleLines"),
            InternetService=json_data.get("InternetService"),
            OnlineSecurity=json_data.get("OnlineSecurity"),
            OnlineBackup=json_data.get("OnlineBackup"),
            DeviceProtection=json_data.get("DeviceProtection"),
            TechSupport=json_data.get("TechSupport"),
            StreamingTV=json_data.get("StreamingTV"),
            StreamingMovies=json_data.get("StreamingMovies"),
            Contract=json_data.get("Contract"),
            PaperlessBilling=json_data.get("PaperlessBilling"),
            PaymentMethod=json_data.get("PaymentMethod"),
            MonthlyCharges=float(json_data.get("MonthlyCharges")),
            TotalCharges=float(json_data.get("TotalCharges"))
        )

        # Get prediction
        final_data = data.get_data_DataFrame()
        pipeline_prediction = PredictionPipeline()
        pred = pipeline_prediction.predict(final_data)

        # Return prediction result
        return jsonify({
            "status": "success",
            "prediction": int(pred[0]),
            "churn_category": "No" if pred[0] == 0 else "Yes"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
