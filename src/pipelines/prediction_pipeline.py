import os
import sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        preprocessor_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
        model_path = os.path.join("artifacts/model_trainer", "model.pkl")

        # Load the preprocessor and model objects
        processor = load_object(preprocessor_path)
        model = load_object(model_path)

        # Transform the input features using the preprocessor
        scaled = processor.transform(features)

        # Predict the outcome using the model
        pred = model.predict(scaled)

        return pred


class CustomClass:
    def __init__(self, 
                 gender: str,
                 SeniorCitizen: int, 
                 Partner: str, 
                 Dependents: str, 
                 tenure: int, 
                 PhoneService: str, 
                 MultipleLines: str,
                 InternetService: str, 
                 OnlineSecurity: str, 
                 OnlineBackup: str, 
                 DeviceProtection: str,
                 TechSupport: str, 
                 StreamingTV: str, 
                 StreamingMovies: str, 
                 Contract: str, 
                 PaperlessBilling: str, 
                 PaymentMethod: str, 
                 MonthlyCharges: float, 
                 TotalCharges: float):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_DataFrame(self):
        try:
            # Create a dictionary with the input data
            custom_input = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }

            # Convert the dictionary into a pandas DataFrame
            data = pd.DataFrame(custom_input)

            return data
        except Exception as e:
            # Raise a custom exception in case of errors
            raise CustomException(e, sys)
