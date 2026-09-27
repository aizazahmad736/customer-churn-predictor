"""
ChurnGuard AI - Inference Engine & Retention Copilot
Provides real-time scoring, risk attribution, personalized retention playbooks,
what-if scenario simulation, and high-throughput batch scoring.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from .config import (
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_COMPARISON_PATH,
    ALL_FEATURE_COLUMNS,
    RETENTION_PLAYBOOKS,
    get_risk_tier,
    ID_COLUMN
)
from .preprocessing import clean_dataset

class ChurnPredictor:
    """Production inference engine for Customer Churn risk scoring."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Run 'python -m src.train' first."
            )
        self.pipeline = joblib.load(model_path)
        self.metrics = self._load_json(METRICS_PATH)
        self.comparison = self._load_json(MODEL_COMPARISON_PATH)
        self.feature_importance = self._load_json(FEATURE_IMPORTANCE_PATH)

    @staticmethod
    def _load_json(path: str) -> Any:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates churn probability and generates personalized risk attribution
        and retention recommendations for a single customer.
        """
        df_input = pd.DataFrame([customer_data])
        df_clean = clean_dataset(df_input)

        # Ensure all required features exist
        for col in ALL_FEATURE_COLUMNS:
            if col not in df_clean.columns:
                df_clean[col] = np.nan

        # Model inference
        proba = float(self.pipeline.predict_proba(df_clean[ALL_FEATURE_COLUMNS])[0, 1])
        tier_info = get_risk_tier(proba)

        # Individual risk factor analysis
        risk_drivers = self._analyze_risk_drivers(customer_data)
        protective_factors = self._analyze_protective_factors(customer_data)
        recommended_actions = self._generate_retention_actions(customer_data, proba)

        return {
            "churn_probability": round(proba, 4),
            "churn_percentage": round(proba * 100.0, 1),
            "predicted_churn": "Yes" if proba >= 0.50 else "No",
            "risk_tier": tier_info["tier"],
            "risk_label": tier_info["label"],
            "color": tier_info["color"],
            "badge": tier_info["badge"],
            "action_urgency": tier_info["action_urgency"],
            "risk_drivers": risk_drivers,
            "protective_factors": protective_factors,
            "recommended_actions": recommended_actions
        }

    def _analyze_risk_drivers(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identifies specific factors pushing customer towards churn."""
        drivers = []
        contract = data.get("Contract", "")
        tenure = float(data.get("tenure", 0))
        payment = data.get("PaymentMethod", "")
        monthly = float(data.get("MonthlyCharges", 0))
        internet = data.get("InternetService", "")
        tech_support = data.get("TechSupport", "")
        security = data.get("OnlineSecurity", "")

        if contract == "Month-to-month":
            drivers.append({
                "factor": "Month-to-Month Contract",
                "severity": "High",
                "detail": "No long-term commitment leaves subscriber vulnerable to rival carrier promotions."
            })
        if tenure <= 6:
            drivers.append({
                "factor": "Early Lifecycle Vulnerability",
                "severity": "High",
                "detail": f"Tenure is only {int(tenure)} month(s). New subscribers exhibit 4x baseline hazard."
            })
        elif tenure <= 18:
            drivers.append({
                "factor": "Sub-2 Year Tenure",
                "severity": "Medium",
                "detail": f"Customer is in year 1-2 ({int(tenure)} mos); brand affinity is still consolidating."
            })
        if payment == "Electronic check":
            drivers.append({
                "factor": "Electronic Check Billing",
                "severity": "High",
                "detail": "Highest attrition payment channel due to repeated manual transaction friction."
            })
        if monthly >= 80:
            drivers.append({
                "factor": "Premium Monthly Pricing",
                "severity": "Medium",
                "detail": f"Monthly fee of ${monthly:.2f} is in the top quartile of billing tiers."
            })
        if internet == "Fiber optic" and tech_support != "Yes":
            drivers.append({
                "factor": "Fiber Optic without Tech Support",
                "severity": "High",
                "detail": "High-bandwidth tier users without dedicated troubleshooting show elevated frustration."
            })
        if security == "No" and internet != "No":
            drivers.append({
                "factor": "Unprotected Internet Connection",
                "severity": "Medium",
                "detail": "Lacks online security package, missing an essential customer retention hook."
            })

        return drivers

    def _analyze_protective_factors(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identifies customer attributes that foster retention and loyalty."""
        protective = []
        contract = data.get("Contract", "")
        tenure = float(data.get("tenure", 0))
        payment = data.get("PaymentMethod", "")
        partner = data.get("Partner", "")
        dependents = data.get("Dependents", "")
        tech_support = data.get("TechSupport", "")

        if contract == "Two year":
            protective.append({
                "factor": "Two-Year Contract Commitment",
                "impact": "Reduces churn propensity by ~80%"
            })
        elif contract == "One year":
            protective.append({
                "factor": "Annual Contract Agreement",
                "impact": "Locks in baseline retention for 12 months"
            })
        if tenure >= 48:
            protective.append({
                "factor": "High Tenure Brand Veteran",
                "impact": f"Loyal relationship over {int(tenure)} months creates high brand inertia"
            })
        if "automatic" in payment.lower():
            protective.append({
                "factor": "Automated Recurring Billing",
                "impact": "Frictionless auto-pay minimizes billing interruption"
            })
        if partner == "Yes" or dependents == "Yes":
            protective.append({
                "factor": "Household Multi-User Account",
                "impact": "Multi-user family accounts have significantly higher switching costs"
            })
        if tech_support == "Yes":
            protective.append({
                "factor": "Active Tech Support Subscription",
                "impact": "Deepens technical engagement and service utility"
            })

        return protective

    def _generate_retention_actions(self, data: Dict[str, Any], proba: float) -> List[Dict[str, str]]:
        """Matches customer risk drivers to business retention playbooks."""
        actions = []
        contract = data.get("Contract", "")
        tenure = float(data.get("tenure", 0))
        payment = data.get("PaymentMethod", "")
        monthly = float(data.get("MonthlyCharges", 0))
        internet = data.get("InternetService", "")
        tech_support = data.get("TechSupport", "")
        security = data.get("OnlineSecurity", "")

        if contract == "Month-to-month":
            actions.append(RETENTION_PLAYBOOKS["contract_month_to_month"])
        if tech_support == "No" and internet != "No":
            actions.append(RETENTION_PLAYBOOKS["no_tech_support"])
        if security == "No" and internet != "No":
            actions.append(RETENTION_PLAYBOOKS["no_online_security"])
        if payment == "Electronic check":
            actions.append(RETENTION_PLAYBOOKS["payment_electronic_check"])
        if monthly >= 80:
            actions.append(RETENTION_PLAYBOOKS["high_monthly_charges"])
        if tenure <= 6:
            actions.append(RETENTION_PLAYBOOKS["low_tenure"])

        # If customer is already low risk and few playbooks match:
        if not actions and proba < 0.30:
            actions.append({
                "title": "Loyalty Appreciation & Cross-Sell Privilege",
                "description": "Customer is highly satisfied and secure. Eligible for VIP anniversary credit or complimentary streaming upgrade.",
                "impact": "Further cements brand evangelism and expands Lifetime Value (CLV)"
            })

        return actions[:3]

    def simulate_what_if(self, base_data: Dict[str, Any], alterations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates hypothetical business interventions to quantify churn probability reduction.
        """
        base_res = self.predict_single(base_data)
        
        # Clone and apply alterations
        mod_data = dict(base_data)
        mod_data.update(alterations)
        
        mod_res = self.predict_single(mod_data)

        delta = mod_res["churn_probability"] - base_res["churn_probability"]
        delta_pct = mod_res["churn_percentage"] - base_res["churn_percentage"]

        return {
            "baseline_probability": base_res["churn_probability"],
            "baseline_percentage": base_res["churn_percentage"],
            "baseline_tier": base_res["risk_tier"],
            "modified_probability": mod_res["churn_probability"],
            "modified_percentage": mod_res["churn_percentage"],
            "modified_tier": mod_res["risk_tier"],
            "delta_percentage": round(delta_pct, 1),
            "improved": delta < 0
        }

    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes high-throughput batch scoring over a DataFrame of customer records.
        """
        df_clean = clean_dataset(df)

        # Validate features
        for col in ALL_FEATURE_COLUMNS:
            if col not in df_clean.columns:
                df_clean[col] = np.nan

        X = df_clean[ALL_FEATURE_COLUMNS]
        probas = self.pipeline.predict_proba(X)[:, 1]

        result_df = df.copy()
        result_df["Churn_Probability"] = [round(float(p), 4) for p in probas]
        result_df["Churn_Percentage"] = [round(float(p) * 100.0, 1) for p in probas]
        result_df["Risk_Tier"] = [get_risk_tier(p)["tier"] for p in probas]
        result_df["Risk_Label"] = [get_risk_tier(p)["label"] for p in probas]
        result_df["Predicted_Churn"] = ["Yes" if p >= 0.50 else "No" for p in probas]

        # Financial impact per record
        monthly_charges = pd.to_numeric(result_df.get("MonthlyCharges", 0), errors="coerce").fillna(0)
        # Revenue at risk = monthly charges if customer is predicted to churn
        result_df["Revenue_At_Risk"] = np.where(result_df["Risk_Tier"] == "HIGH", monthly_charges, 0.0)

        # Summary KPIs
        total_customers = len(result_df)
        high_risk_count = int((result_df["Risk_Tier"] == "HIGH").sum())
        mod_risk_count = int((result_df["Risk_Tier"] == "MODERATE").sum())
        low_risk_count = int((result_df["Risk_Tier"] == "LOW").sum())
        predicted_churners = int((result_df["Predicted_Churn"] == "Yes").sum())
        total_revenue_at_risk = float(result_df["Revenue_At_Risk"].sum())
        avg_churn_risk = float(result_df["Churn_Probability"].mean() * 100.0)

        summary = {
            "total_customers": total_customers,
            "predicted_churners": predicted_churners,
            "churn_rate_pct": round((predicted_churners / total_customers * 100.0) if total_customers > 0 else 0, 1),
            "high_risk_count": high_risk_count,
            "moderate_risk_count": mod_risk_count,
            "low_risk_count": low_risk_count,
            "total_revenue_at_risk_monthly": round(total_revenue_at_risk, 2),
            "average_churn_probability": round(avg_churn_risk, 1)
        }

        return result_df, summary
