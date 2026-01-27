import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
    )

    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['charges_per_tenure'] = df['MonthlyCharges'] * df['tenure']
    df['charge_difference'] = df['TotalCharges'] - df['charges_per_tenure']

    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    def count_services(row):
        count = 0
        for col in service_cols:
            if col in row.index and row[col] not in ['No', 'No phone service', 'No internet service']:
                count += 1
        return count

    df['total_services'] = df.apply(count_services, axis=1)

    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df['contract_months'] = df['Contract'].map(contract_map)
    df['contract_value'] = df['MonthlyCharges'] * df['contract_months']

    df['has_partner_dependents'] = ((df['Partner'] == 'Yes') & (df['Dependents'] == 'Yes')).astype(int)
    df['is_senior_alone'] = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'No')).astype(int)

    df['has_internet'] = (df['InternetService'] != 'No').astype(int)
    df['has_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)

    df['no_protection'] = (
        (df['OnlineSecurity'] == 'No') & (df['OnlineBackup'] == 'No') &
        (df['DeviceProtection'] == 'No') & (df['TechSupport'] == 'No')
    ).astype(int)

    df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)

    df['auto_payment'] = df['PaymentMethod'].isin([
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ]).astype(int)

    df['monthly_charges_cat'] = pd.cut(
        df['MonthlyCharges'],
        bins=[0, 35, 70, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    return df


def get_feature_columns(include_engineered=True):
    numerical = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    categorical = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    if include_engineered:
        numerical.extend([
            'avg_monthly_charges', 'charges_per_tenure', 'charge_difference',
            'total_services', 'contract_months', 'contract_value',
            'has_partner_dependents', 'is_senior_alone', 'has_internet',
            'has_fiber', 'no_protection', 'has_streaming', 'auto_payment'
        ])
        categorical.extend(['tenure_group', 'monthly_charges_cat'])

    return {'numerical': numerical, 'categorical': categorical}
