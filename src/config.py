"""
ChurnGuard AI - System Configuration
Centralized definitions for schema, feature groupings, risk thresholds,
paths, and business retention playbooks.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "telco_customer_churn.csv")
SAMPLE_BATCH_PATH = os.path.join(DATA_DIR, "sample_batch_customers.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.json")
MODEL_COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

# Schema definitions
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Risk classification thresholds
RISK_THRESHOLDS = {
    "LOW": 0.30,
    "MODERATE": 0.65
}

def get_risk_tier(probability: float) -> dict:
    """Returns risk tier descriptor and UI style metadata."""
    if probability < RISK_THRESHOLDS["LOW"]:
        return {
            "tier": "LOW",
            "label": "Low Risk (Retained / Loyal)",
            "color": "#10B981", # Emerald
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "border_color": "#10B981",
            "badge": "🟢 LOW RISK",
            "action_urgency": "Standard Account Maintenance"
        }
    elif probability < RISK_THRESHOLDS["MODERATE"]:
        return {
            "tier": "MODERATE",
            "label": "Moderate Risk (Watchlist)",
            "color": "#F59E0B", # Amber
            "bg_color": "rgba(245, 158, 11, 0.15)",
            "border_color": "#F59E0B",
            "badge": "🟡 MODERATE RISK",
            "action_urgency": "Proactive Engagement Required"
        }
    else:
        return {
            "tier": "HIGH",
            "label": "Critical Risk (Likely to Churn)",
            "color": "#EF4444", # Crimson
            "bg_color": "rgba(239, 68, 68, 0.15)",
            "border_color": "#EF4444",
            "badge": "🔴 CRITICAL RISK",
            "action_urgency": "Immediate Retention Intervention"
        }

# Strategic business retention recommendations tailored to individual customer signals
RETENTION_PLAYBOOKS = {
    "contract_month_to_month": {
        "title": "Contract Lock-In Incentive",
        "description": "Customer is on Month-to-Month contract. Offer an exclusive 15% discount or billing credit to migrate to an Annual or Two-Year commitment plan.",
        "impact": "Reduces churn hazard by up to 55%"
    },
    "no_tech_support": {
        "title": "Complimentary Premium Tech Support Bundle",
        "description": "Customer has no Tech Support. Activate 3 months of free 24/7 dedicated support and device diagnostics.",
        "impact": "Increases customer satisfaction and sticky service adoption by 35%"
    },
    "no_online_security": {
        "title": "Security & Identity Protection Upgrade",
        "description": "Customer lacks Online Security. Provide complimentary anti-virus/anti-phishing bundle for all connected devices.",
        "impact": "Lowers churn propensity by 28%"
    },
    "payment_electronic_check": {
        "title": "Auto-Pay Incentive Credit",
        "description": "Electronic check payments have highest billing friction. Offer a $10 one-time account credit for enrolling in Automatic Credit Card or Bank Transfer billing.",
        "impact": "Eliminates monthly payment friction and bill-shock drop-off"
    },
    "high_monthly_charges": {
        "title": "Personalized Value Plan Optimization",
        "description": "Customer's monthly fee exceeds $80/mo. Review active usage and offer a customized bundle with high-speed fidelity at optimized tier pricing.",
        "impact": "Preserves core revenue while eliminating price-sensitivity churn"
    },
    "low_tenure": {
        "title": "VIP Onboarding & Dedicated Success Outreach",
        "description": "Customer is in the vulnerable initial 6-month lifecycle. Schedule proactive check-in call with a Senior Customer Success Specialist.",
        "impact": "Triples 1-year retention probability for early-stage subscribers"
    }
}
