"""
ChurnGuard AI - Telco Churn Dataset Generator
Generates realistic, statistically calibrated customer data modeled after the
benchmark Telco Customer Churn dataset with ground-truth business patterns.
"""

import os
import random
import numpy as np
import pandas as pd

def generate_telco_churn_dataset(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic Telco customer churn dataset with known
    domain dynamics:
      - Month-to-month contracts have substantially higher churn.
      - Low tenure (< 12 months) has the highest hazard rate.
      - Fiber optic users with high monthly charges and no tech support churn frequently.
      - Long-tenure, multi-service, automated payment customers have high loyalty (< 5% churn).
    """
    np.random.seed(random_state)
    random.seed(random_state)

    customer_ids = [f"{random.randint(1000, 9999)}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}" for _ in range(n_samples)]
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.505, 0.495])
    senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partners = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    
    # Dependents correlate with Partner
    dependents = []
    for p in partners:
        if p == "Yes":
            dependents.append(np.random.choice(["Yes", "No"], p=[0.51, 0.49]))
        else:
            dependents.append(np.random.choice(["Yes", "No"], p=[0.12, 0.88]))
    dependents = np.array(dependents)

    # Tenure in months (bimodal: heavy at 1-6 months and 60-72 months)
    tenure_raw = np.concatenate([
        np.random.exponential(scale=12, size=int(n_samples * 0.45)),
        np.random.uniform(1, 72, size=int(n_samples * 0.30)),
        np.random.normal(loc=65, scale=5, size=n_samples - int(n_samples * 0.45) - int(n_samples * 0.30))
    ])
    np.random.shuffle(tenure_raw)
    tenures = np.clip(np.round(tenure_raw), 1, 72).astype(int)

    # Phone & Multiple lines
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.46, 0.54]))

    # Internet Service
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])

    # Add-on services
    def sample_service(internet_arr, prob_yes):
        res = []
        for i_srv in internet_arr:
            if i_srv == "No":
                res.append("No internet service")
            else:
                res.append(np.random.choice(["Yes", "No"], p=[prob_yes, 1.0 - prob_yes]))
        return res

    online_security = sample_service(internet_service, prob_yes=0.36)
    online_backup = sample_service(internet_service, prob_yes=0.44)
    device_protection = sample_service(internet_service, prob_yes=0.43)
    tech_support = sample_service(internet_service, prob_yes=0.37)
    streaming_tv = sample_service(internet_service, prob_yes=0.49)
    streaming_movies = sample_service(internet_service, prob_yes=0.50)

    # Contract
    contracts = []
    for t in tenures:
        if t < 12:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.88, 0.10, 0.02]))
        elif t < 36:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.45, 0.40, 0.15]))
        else:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.18, 0.35, 0.47]))

    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])

    payment_methods = np.random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ], size=n_samples, p=[0.34, 0.23, 0.22, 0.21])

    # Monthly Charges calculation based on subscribed services
    monthly_charges = []
    for i in range(n_samples):
        charge = 20.0  # Base line charge
        if phone_service[i] == "Yes":
            charge += 10.0
            if multiple_lines[i] == "Yes":
                charge += 12.0
        if internet_service[i] == "DSL":
            charge += 25.0
        elif internet_service[i] == "Fiber optic":
            charge += 45.0
        
        # Add-ons
        for srv in [online_security[i], online_backup[i], device_protection[i], tech_support[i]]:
            if srv == "Yes":
                charge += 6.5
        for stm in [streaming_tv[i], streaming_movies[i]]:
            if stm == "Yes":
                charge += 10.0
        
        # Add realistic noise
        charge += np.random.normal(0, 3.5)
        monthly_charges.append(round(max(18.25, min(118.75, charge)), 2))
    
    monthly_charges = np.array(monthly_charges)

    # Total Charges
    total_charges = []
    for i in range(n_samples):
        # tenure * monthly_charges with minor variance
        tc = tenures[i] * monthly_charges[i] * np.random.uniform(0.95, 1.05)
        total_charges.append(round(tc, 2))
    total_charges = np.array(total_charges)

    # Realistic Churn Probability Model (Log-Odds formulation)
    churn_labels = []
    for i in range(n_samples):
        # Base log-odds
        logit = -1.6
        
        # Tenure impact (exponential decay in churn risk)
        logit -= 0.05 * tenures[i]
        
        # Contract impact
        if contracts[i] == "Month-to-month":
            logit += 1.45
        elif contracts[i] == "Two year":
            logit -= 1.35
        elif contracts[i] == "One year":
            logit -= 0.55
            
        # Internet Service impact
        if internet_service[i] == "Fiber optic":
            logit += 0.75
        elif internet_service[i] == "No":
            logit -= 0.65
            
        # Tech support & Security impact
        if tech_support[i] == "Yes":
            logit -= 0.55
        if online_security[i] == "Yes":
            logit -= 0.50
            
        # Payment method
        if payment_methods[i] == "Electronic check":
            logit += 0.65
        elif "automatic" in payment_methods[i]:
            logit -= 0.40
            
        # Senior citizen & Dependents
        if senior_citizens[i] == 1:
            logit += 0.25
        if dependents[i] == "Yes":
            logit -= 0.30
            
        # Monthly charge impact
        if monthly_charges[i] > 80:
            logit += 0.45
        elif monthly_charges[i] < 35:
            logit -= 0.35
            
        prob = 1.0 / (1.0 + np.exp(-logit))
        churn = "Yes" if np.random.random() < prob else "No"
        churn_labels.append(churn)

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenures,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels
    })

    return df

def generate_sample_batch_file(full_df: pd.DataFrame, out_path: str, n_samples: int = 20):
    """Generates a diverse 20-customer batch test file with varying risk profiles."""
    sample_df = full_df.sample(n=n_samples, random_state=123).copy()
    # In batch files, Churn column is typically what we want to predict (omitted or kept as benchmark)
    # We provide a clean format suitable for prediction
    sample_df = sample_df.drop(columns=["Churn"])
    sample_df.to_csv(out_path, index=False)
    print(f"Created sample batch file with {len(sample_df)} records: {out_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "telco_customer_churn.csv")
    sample_path = os.path.join(base_dir, "sample_batch_customers.csv")
    
    print("Generating comprehensive Telco Customer Churn dataset...")
    df = generate_telco_churn_dataset(n_samples=7043, random_state=42)
    df.to_csv(data_path, index=False)
    print(f"Successfully saved {len(df)} records to: {data_path}")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")
    
    generate_sample_batch_file(df, sample_path, n_samples=20)
