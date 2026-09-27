"""
ChurnGuard AI - Automated Test Suite
Verifies data pipeline, ML inference, risk tiering, what-if simulations, and batch scoring.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.config import (
    DATASET_PATH,
    SAMPLE_BATCH_PATH,
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.preprocessing import clean_dataset, build_preprocessor
from src.predictor import ChurnPredictor

class TestChurnGuardPipeline(unittest.TestCase):
    """Unit and integration test cases for ChurnGuard AI."""

    @classmethod
    def setUpClass(cls):
        """Initializes predictor instance once for test suite."""
        cls.predictor = ChurnPredictor()

    def test_dataset_exists_and_schema_valid(self):
        """Verify the Telco dataset exists and matches expected schema."""
        self.assertTrue(os.path.exists(DATASET_PATH), f"Dataset missing at {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        self.assertGreater(len(df), 5000)
        
        for col in ALL_FEATURE_COLUMNS:
            self.assertIn(col, df.columns, f"Feature column missing: {col}")
        self.assertIn("Churn", df.columns)

    def test_data_cleaning_handles_missing_total_charges(self):
        """Verify cleaning correctly computes TotalCharges when missing or string."""
        raw_df = pd.DataFrame([{
            "tenure": 10,
            "MonthlyCharges": 50.0,
            "TotalCharges": " " # Blank string
        }])
        clean_df = clean_dataset(raw_df)
        self.assertAlmostEqual(clean_df["TotalCharges"].iloc[0], 500.0)

    def test_single_prediction_returns_valid_structure(self):
        """Verify single customer prediction returns valid probability and keys."""
        test_customer = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.50,
            "TotalCharges": 286.50
        }
        res = self.predictor.predict_single(test_customer)
        self.assertIn("churn_probability", res)
        self.assertGreaterEqual(res["churn_probability"], 0.0)
        self.assertLessEqual(res["churn_probability"], 1.0)
        self.assertIn(res["risk_tier"], ["LOW", "MODERATE", "HIGH"])
        self.assertTrue(len(res["risk_drivers"]) > 0)
        self.assertTrue(len(res["recommended_actions"]) > 0)

    def test_risk_contrast_high_vs_low(self):
        """Verify that a high-risk profile scores higher churn probability than a loyal profile."""
        high_risk = {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0
        }

        loyal_customer = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 65,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 45.0,
            "TotalCharges": 2925.0
        }

        high_res = self.predictor.predict_single(high_risk)
        loyal_res = self.predictor.predict_single(loyal_customer)

        self.assertGreater(high_res["churn_probability"], loyal_res["churn_probability"])
        self.assertEqual(loyal_res["risk_tier"], "LOW")

    def test_what_if_simulation_reduces_churn(self):
        """Verify what-if simulation calculates expected reduction when upgrading contract."""
        base = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 4,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 90.0,
            "TotalCharges": 360.0
        }
        sim_res = self.predictor.simulate_what_if(base, {"Contract": "Two year", "TechSupport": "Yes"})
        self.assertTrue(sim_res["improved"])
        self.assertLess(sim_res["modified_probability"], sim_res["baseline_probability"])
        self.assertLess(sim_res["delta_percentage"], 0)

    def test_batch_prediction_pipeline(self):
        """Verify batch prediction processes DataFrame and generates correct summary metrics."""
        self.assertTrue(os.path.exists(SAMPLE_BATCH_PATH))
        df_batch = pd.read_csv(SAMPLE_BATCH_PATH)
        scored_df, summary = self.predictor.predict_batch(df_batch)

        self.assertEqual(len(scored_df), len(df_batch))
        self.assertIn("Churn_Probability", scored_df.columns)
        self.assertIn("Risk_Tier", scored_df.columns)
        self.assertIn("Revenue_At_Risk", scored_df.columns)
        self.assertEqual(summary["total_customers"], len(df_batch))
        self.assertGreaterEqual(summary["high_risk_count"], 0)

if __name__ == "__main__":
    unittest.main()
