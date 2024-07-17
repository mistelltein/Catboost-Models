import joblib
import pandas as pd

model = joblib.load('catboost_credit_approval_model.pkl')

def preprocess_new_data(data, categorical_features):
    for col in categorical_features:
        data[col] = data[col].astype(str)
    X_new = data.drop(columns=['id'])
    return X_new

new_data = pd.DataFrame({
    'id': [1],
    'age': [54],
    'gender': ['female'],
    'annual_income': [23251],
    'marital_status': ['divorced'],
    'num_children': [4],
    'education': ['high_school'],
    'years_of_employment': [2],
    'credit_history': ['poor'],
    'current_debt': [19330],
    'loan_purpose': ['business']
})

categorical_features = ['age', 'gender', 'annual_income', 'marital_status', 'education', 'years_of_employment', 'credit_history', 'current_debt', 'loan_purpose']

X_new = preprocess_new_data(new_data, categorical_features)

new_prediction = model.predict(X_new)
prediction_result = new_prediction[0] == 1

print(f'The prediction for the new data is: {prediction_result}')
