# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Reading in the data
data = pd.read_csv("car_data.csv")
print("First 5 rows of data:")
print(data.head(5))

# Feature lists
categorical = ['type', 'drive', 'make', 'model','transmission']
numerical = ['cylinders', 'displacement']
output = 'combination_mpg' 
categorical_for_model = ['type', 'drive', 'fuel_type', 'make', 'transmission']

# Data preparation
data = data.dropna(subset=[output])
for col in numerical:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mean())
for col in categorical_for_model:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mode()[0])

# Defining feature and target 
feature_cols = categorical_for_model + numerical
X = data[feature_cols]
y = data[output]

print("\nFeature columns used for the model:")
print(feature_cols)
print(f"\nNumber of samples after cleaning: {len(X)}")

# One-hot encoding and linear regression
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_for_model)
    ],
    remainder='passthrough'  # keep numerical columns as-is
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} samples")
print(f"Test size: {X_test.shape[0]} samples")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nLinear Regression R² Score on test set: {r2:.3f}")

# Graphing stuff
viz_feature = 'displacement'
X_viz = data[[viz_feature]].values
y_viz = data[output].values
simple_lr = LinearRegression()
simple_lr.fit(X_viz, y_viz)
x_line = np.linspace(X_viz.min(), X_viz.max(), 200).reshape(-1, 1)
y_line = simple_lr.predict(x_line)
slope = simple_lr.coef_[0]
intercept = simple_lr.intercept_
eq_text = f"MPG = {slope:.3f} × {viz_feature} + {intercept:.3f}"
plt.figure(figsize=(8, 5))
plt.scatter(X_viz, y_viz, alpha=0.4, label='Actual Data')
plt.plot(x_line, y_line, linewidth=2, label='Trend Line')
plt.title(f"Fuel Efficiency vs. {viz_feature.capitalize()}")
plt.xlabel(viz_feature.capitalize())
plt.ylabel("Combined MPG")
plt.legend()
plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.7))

plt.tight_layout()
plt.show()

# Categorical feature exploration
for col in categorical:
    print(f"\n{col}: {data[col].nunique()} unique classes")
    print(data[col].value_counts().head(10))
for col in categorical:
    plt.figure(figsize=(8, 4))
    data[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
plt.show()