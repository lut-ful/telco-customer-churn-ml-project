import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from feast import Field, FeatureStore, Entity, FeatureView, FileSource
from feast.types import Int64, String, Float32
from feast.value_type import ValueType
from datetime import datetime, timedelta

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
    feature_store_repo_path = "feature_repo"

class DataTransformation:
    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
            
            # Get absolute path and create directory structure
            repo_path = os.path.abspath(self.data_transformation_config.feature_store_repo_path)
            os.makedirs(os.path.join(repo_path, "data"), exist_ok=True)
            
            # Create feature store yaml with minimal configuration
            feature_store_yaml_path = os.path.join(repo_path, "feature_store.yaml")
            
            # Simplified, minimal feature store configuration
            feature_store_yaml = """project: Churn_Prediction
provider: local
registry: data/registry.db
online_store:
  type: sqlite
offline_store:
  type: file
entity_key_serialization_version: 2"""
            
            # Write configuration file
            with open(feature_store_yaml_path, 'w') as f:
                f.write(feature_store_yaml)
            
            logging.info(f"Created feature store configuration at {feature_store_yaml_path}")
            
            # Verify the configuration file content
            with open(feature_store_yaml_path, 'r') as f:
                logging.info(f"Configuration file content:\n{f.read()}")
            
            # Initialize feature store
            self.feature_store = FeatureStore(repo_path=repo_path)
            logging.info("Feature store initialized successfully")

        except Exception as e:
            logging.error(f"Error in initialization: {str(e)}")
            raise CustomException(e, sys)

    def get_data_transformation_obj(self,numerical_features,categorical_features):
        try:
            logging.info("Data Transformation Started")

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            from sklearn.preprocessing import OneHotEncoder

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder())
            ])

            # Combine numerical and categorical pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR
            
            df.loc[(df[col] > upper_limit), col] = upper_limit
            df.loc[(df[col] < lower_limit), col] = lower_limit
            
            return df

        except Exception as e:
            logging.info("Outliers handling code")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")


            target_column_name = "Churn"
            logging.info(f"target Column {target_column_name}")
            numerical_columns = train_data.select_dtypes(exclude='object').columns.tolist()
            categorical_columns = train_data.select_dtypes(include='object').columns.tolist()
            if target_column_name in numerical_columns:
                numerical_columns.remove(target_column_name)
            logging.info(f"Numerical Columns {numerical_columns}")
            
            if target_column_name in categorical_columns:
                categorical_columns.remove(target_column_name)
            logging.info(f"Categorical Columns {categorical_columns}")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_obj(numerical_columns,categorical_columns)

            input_feature_train_df = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_data[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Starting feature store operations")
            
            # Push data to Feast feature store
            self.push_features_to_store(train_data, "train")
            logging.info("Pushed training data to feature store")
            
            self.push_features_to_store(test_data, "test")
            logging.info("Pushed testing data to feature store")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocess_obj_file_path

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

    def push_features_to_store(self, df, entity_id):
        try:
            # Add timestamp column if not present
            if 'event_timestamp' not in df.columns:
                df['event_timestamp'] = pd.Timestamp.now()
            
            # Add entity_id column if not present
            if 'entity_id' not in df.columns:
                df['entity_id'] = range(len(df))

            # Save data as parquet
            data_path = os.path.join(
                self.data_transformation_config.feature_store_repo_path,
                "data"
            )
            parquet_path = os.path.join(data_path, f"{entity_id}_features.parquet")
            
            # Ensure the directory exists
            os.makedirs(data_path, exist_ok=True)
            
            # Save the parquet file
            df.to_parquet(parquet_path, index=False)
            logging.info(f"Saved feature data to {parquet_path}")

            # Define data source with relative path
            data_source = FileSource(
                path=f"data/{entity_id}_features.parquet",
                timestamp_field="event_timestamp"
            )

            # Define entity
            entity = Entity(
                name="entity_id",
                value_type=ValueType.INT64,
                description="Entity ID"
            )
            # Define feature view
            feature_view = FeatureView(
                name=f"{entity_id}_features",
                entities=[entity],
                schema=[
                    Field(name="gender", dtype=String),
                    Field(name="SeniorCitizen", dtype=Int64),
                    Field(name="Partner", dtype=String),
                    Field(name="Dependents", dtype=String),
                    Field(name="tenure", dtype=Int64),
                    Field(name="PhoneService", dtype=String),
                    Field(name="MultipleLines", dtype=String),
                    Field(name="InternetService", dtype=String),
                    Field(name="OnlineSecurity", dtype=String),
                    Field(name="OnlineBackup", dtype=String),
                    Field(name="DeviceProtection", dtype=String),
                    Field(name="TechSupport", dtype=String),
                    Field(name="StreamingTV", dtype=String),
                    Field(name="StreamingMovies", dtype=String),
                    Field(name="Contract", dtype=String),
                    Field(name="PaperlessBilling", dtype=String),
                    Field(name="PaymentMethod", dtype=String),
                    Field(name="MonthlyCharges", dtype=Float32),
                    Field(name="TotalCharges", dtype=Float32)
                ],
                source=data_source,
                online=True
            )

            # Apply to feature store
            self.feature_store.apply([entity, feature_view])
            logging.info(f"Applied entity and feature view for {entity_id}")

            # Materialize features
            self.feature_store.materialize(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now() + timedelta(days=1)
            )
            logging.info("Materialized features successfully")

        except Exception as e:
            logging.error(f"Error in push_features_to_store: {str(e)}")
            raise CustomException(e, sys)

    def retrieve_features_from_store(self, entity_id):
        try:
            feature_service_name = f"{entity_id}_features"
            feature_vector = self.feature_store.get_online_features(
                feature_refs=[
                    f"{entity_id}_features:gender",
                    f"{entity_id}_features:SeniorCitizen",
                    f"{entity_id}_features:Partner",
                    f"{entity_id}_features:Dependents",
                    f"{entity_id}_features:tenure",
                    f"{entity_id}_features:PhoneService",
                    f"{entity_id}_features:MultipleLines",
                    f"{entity_id}_features:InternetService",
                    f"{entity_id}_features:OnlineSecurity",
                    f"{entity_id}_features:OnlineBackup",
                    f"{entity_id}_features:DeviceProtection",
                    f"{entity_id}_features:TechSupport",
                    f"{entity_id}_features:StreamingTV",
                    f"{entity_id}_features:StreamingMovies",
                    f"{entity_id}_features:Contract",
                    f"{entity_id}_features:PaperlessBilling",
                    f"{entity_id}_features:PaymentMethod",
                    f"{entity_id}_features:MonthlyCharges",
                    f"{entity_id}_features:TotalCharges"
                ],
                entity_rows=[{"entity_id": i} for i in range(len(df))]
            ).to_df()

            logging.info(f"Retrieved features for {entity_id}")
            return feature_vector

        except Exception as e:
            logging.error(f"Error in retrieve_features_from_store: {str(e)}")
            raise CustomException(e, sys)