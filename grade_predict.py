import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_file = 'Synthetic Data/Grades/student_grades_kyrgyz_80k.csv'
test_file = 'Synthetic Data/Grades/student_grades_kyrgyz_20k.csv'

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

train_data['gender'] = train_data['gender'].map({'male': 0, 'female': 1})
train_data['attendance'] = train_data['attendance'].map({'low': 0, 'medium': 1, 'high': 2})
train_data['participation'] = train_data['participation'].map({'low': 0, 'medium': 1, 'high': 2})

test_data['gender'] = test_data['gender'].map({'male': 0, 'female': 1})
test_data['attendance'] = test_data['attendance'].map({'low': 0, 'medium': 1, 'high': 2})
test_data['participation'] = test_data['participation'].map({'low': 0, 'medium': 1, 'high': 2})

features = ['age', 'gender', 'math_grade', 'literature_grade', 'attendance', 'participation', 'homework',
            'midterm_grade']
target = 'final_grade'

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

model = CatBoostRegressor(
    depth=9,
    iterations=1000,
    l2_leaf_reg=9,
    learning_rate=0.01,
    verbose=100
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


def accuracy_with_tolerance(y_true, y_pred, tolerance=5):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= tolerance:
            correct += 1
    return correct / len(y_true)


tolerance = 5
accuracy = accuracy_with_tolerance(y_test, y_pred, tolerance)

print(f'Accuracy within ±{tolerance} tolerance: {accuracy * 100:.2f}%')

feature_importances = model.get_feature_importance(Pool(X_train, label=y_train))
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

feature_importance_text = "\n".join(
    [f"Feature: {name}, Importance: {importance}" for name, importance in zip(feature_names, feature_importances)])
print(feature_importance_text)

output = f"Mean Absolute Error: {mae}\nMean Squared Error: {mse}\nR^2 Score: {r2}\n\nAccuracy within ±{tolerance} tolerance: {accuracy * 100:.2f}%\n\n{feature_importance_text}"
print(output)