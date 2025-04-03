import pandas as pd
import numpy as np



# 1. Data Loading and Preprocessing
def load_and_preprocess_data(filepath):
    """
    Load and preprocess the diabetes dataset
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(filepath)

    # Basic info
    print(f"Dataset shape: {df.shape}")

    # Define target variable
    # Convert readmitted to binary classification: >30 and NO to 0, <30 to 1
    df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 0, '<30': 1})

    # Check class distribution
    print("Class distribution:")
    print(df['readmitted'].value_counts(normalize=True))

    # Handle missing values
    df.replace('?', np.nan, inplace=True)

    # Identify categorical and numerical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols.remove('readmitted')  # Remove target from numerical columns

    # Feature Engineering

    # Group diagnosis codes
    def map_diagnosis(diag):
        if pd.isna(diag) or diag == '?':
            return 'Unknown'
        try:
            diag_float = float(diag)
            if 390 <= diag_float <= 459 or diag_float == 785:
                return 'Circulatory'
            elif 460 <= diag_float <= 519 or diag_float == 786:
                return 'Respiratory'
            elif 520 <= diag_float <= 579 or diag_float == 787:
                return 'Digestive'
            elif 250 <= diag_float <= 250.99:
                return 'Diabetes'
            elif 800 <= diag_float <= 999:
                return 'Injury'
            elif 710 <= diag_float <= 739:
                return 'Musculoskeletal'
            elif 580 <= diag_float <= 629 or diag_float == 788:
                return 'Genitourinary'
            elif 140 <= diag_float <= 239:
                return 'Neoplasms'
            else:
                return 'Other'
        except:
            return 'Other'

    df['diag_1_group'] = df['diag_1'].apply(map_diagnosis)
    df['diag_2_group'] = df['diag_2'].apply(map_diagnosis)
    df['diag_3_group'] = df['diag_3'].apply(map_diagnosis)

    # Create feature for number of medications
    med_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                  'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                  'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                  'miglitol', 'troglitazone', 'tolazamide', 'examide',
                  'citoglipton', 'insulin', 'glyburide-metformin',
                  'glipizide-metformin', 'glimepiride-pioglitazone',
                  'metformin-rosiglitazone', 'metformin-pioglitazone']

    # Convert medication columns to binary (1 if any medication was given, 0 if not)
    for col in med_columns:
        df[col] = df[col].map({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1})

    # Calculate total medications count
    df['total_med_taken'] = df[med_columns].sum(axis=1)

    # Age as numerical
    age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
               '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
               '[80-90)': 85, '[90-100)': 95}
    df['age_num'] = df['age'].map(age_map)

    # Add features for medical complexity
    df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['medical_complexity'] = df['num_lab_procedures'] + df['num_procedures']

    # Select final features
    cat_features = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id',
                   'admission_source_id', 'medical_specialty', 'diag_1_group',
                   'diag_2_group', 'diag_3_group', 'max_glu_serum', 'A1Cresult',
                   'change', 'diabetesMed']

    num_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency',
                   'number_inpatient', 'number_diagnoses', 'age_num',
                   'total_med_taken', 'service_utilization', 'medical_complexity']

    # Create X and y
    X = df[cat_features + num_features]
    y = df['readmitted']

    return X, y, cat_features, num_features