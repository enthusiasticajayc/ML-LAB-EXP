
import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination

# Read Cleveland Heart Disease data
heartDisease = pd.read_csv('heart.csv')

# Replace missing-value markers
heartDisease = heartDisease.replace('?', np.nan)

# Rename target column to heartdisease
if 'target' in heartDisease.columns and 'heartdisease' not in heartDisease.columns:
    heartDisease.rename(
        columns={'target': 'heartdisease'},
        inplace=True
    )

# Display sample data
print('Sample instances from the dataset are given below:')
print(heartDisease.head())

# Display attributes and data types
print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# Create Bayesian Network
model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

# Ensure all model columns exist
required_columns = [
    'age', 'sex', 'exang', 'cp',
    'heartdisease', 'restecg', 'chol'
]

missing_columns = [
    col for col in required_columns
    if col not in heartDisease.columns
]

if missing_columns:
    raise ValueError(
        f'Missing columns in heart.csv: {missing_columns}'
    )

# Keep required columns and remove incomplete rows
data = heartDisease[required_columns].dropna()

# Convert selected columns to integers
data = data.astype(int)

# Learn CPDs using Maximum Likelihood Estimation
print('\nLearning CPDs using Maximum Likelihood Estimation')

model.fit(data, estimator=DiscreteMLE())

# Perform inference
print('\nInferencing with Bayesian Network:')
heartDisease_infer = VariableElimination(model)

# Query 1: Heart disease given restecg = 1
print('\n1. Probability of heartdisease given evidence restecg = 1:')

q1 = heartDisease_infer.query(
    variables=['heartdisease'],
    evidence={'restecg': 1}
)
print(q1)

# Query 2: Heart disease given cp = 2
print('\n2. Probability of heartdisease given evidence cp = 2:')

q2 = heartDisease_infer.query(
    variables=['heartdisease'],
    evidence={'cp': 2}
)
print(q2)
