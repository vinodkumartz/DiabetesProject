# Import libraries
import pandas as pd
import sys
import os

# Get the current script directory
cur_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory
root_dir = os.path.abspath(os.path.join(cur_dir, "../../"))

# If needed for module imports (e.g., custom utilities)
# sys.path.append(root_dir)  # <- Keep this only if importing from `root_dir`

# Function to assign correct dtypes
def add_dtypes(data):
    '''Returns data with dtypes correctly assigned.'''
    num_cols = ['time_in_hospital', 'num_lab_procedures',
                'num_procedures', 'num_medications', 'number_outpatient',
                'number_emergency', 'number_inpatient', 'number_diagnoses']
    
    numeric_cols = [x for x in num_cols if x in data.columns]  
    categorical = data.columns.difference(numeric_cols)

    data[numeric_cols] = data[numeric_cols].astype('float')
    data[categorical] = data[categorical].astype('object')

    return data




def load_data(processed=True, weight=False):
    
    '''Returns the data.
    processed: bool: type of data to load (raw or processed)
    weight: bool: whether to include weight in the data
    '''

    if not processed:
        file_path = os.path.join(root_dir, r'data/raw/diabetes.csv')
    else:
        if weight:
            file_path = os.path.join(root_dir, r'data/preprocessed/diabetes_with_weight_cleaned.csv')
        else:
            file_path = os.path.join(root_dir, r'data/preprocessed/diabetes_without_weight_cleaned.csv')

    file_path = os.path.abspath(file_path)

    print("Resolved file path:", file_path)

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return None  # Or handle it gracefully

    data = pd.read_csv(file_path)
    return add_dtypes(data)
