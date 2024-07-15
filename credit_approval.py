import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report


train_good = pd.read_csv('Synthetic Data/Bank/good_credit_data_100k.csv')
train_poor = pd.read_csv('Synthetic Data/Bank/poor_credit_data_100k.csv')
test_good = pd.read_csv('Synthetic Data/Bank/good_credit_data_25k.csv')
test_poor = pd.read_csv('Synthetic Data/Bank/poor_credit_data_25k.csv')

train_good['target'] = 1
train_poor['target'] = 0
test_good['target'] = 1
test_poor['target'] = 0

train_data = pd.concat([train_good, train_poor], axis=0)
test_data = pd.concat([test_good, test_poor], axis=0)

categorical_features = ['age', 'gender', 'annual_income', 'marital_status', 'education', 'years_of_employment', 'current_debt', 'credit_history', 'loan_purpose']

for col in categorical_features:
    train_data[col] = train_data[col].astype(str)
    test_data[col] = test_data[col].astype(str)

X_train = train_data.drop(columns=['id', 'target'])
y_train = train_data['target']
X_test = test_data.drop(columns=['id', 'target'])
y_test = test_data['target']

categorical_features_indices = [X_train.columns.get_loc(col) for col in categorical_features]

model = CatBoostClassifier(
    depth=9,
    iterations=1000,
    l2_leaf_reg=9,
    learning_rate=0.01,
    verbose=100
)

model.fit(X_train, y_train, cat_features=categorical_features_indices)

feature_importances = model.get_feature_importance(Pool(X_train, label=y_train, cat_features=categorical_features_indices))
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

feature_importance_text = "\n".join([f"Feature: {name}, Importance: {importance}" for name, importance in zip(feature_names, feature_importances)])
print(feature_importance_text)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred)
output = f"Accuracy: {accuracy}\n\n{report}\n\n{feature_importance_text}"
print(output)
