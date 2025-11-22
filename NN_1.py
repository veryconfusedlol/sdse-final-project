# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 17:25:43 2025

@author: laley
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Reading in the data
data = pd.read_csv("car_data.csv")
# print("First 5 rows of data:")
# print(data.head(5))

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



# Testing out NN
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
n_features = X_train_transformed.shape[1]

model_NN1 = Sequential()
model_NN1.add(Dense(5,activation='sigmoid',input_shape=(n_features,)))
model_NN1.add(Dense(4,activation='relu'))
model_NN1.add(Dense(3,activation='tanh'))
model_NN1.add(Dense(1))
model_NN1.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])
model_NN1.fit(X_train_transformed,y_train, batch_size=None, epochs=1,
              validation_split=0.0,validation_data=None)
test_performance_NN1 = model_NN1.evaluate(X_test_transformed,y_test)


# Testing out NN2
model_NN2 = Sequential()
model_NN2.add(Dense(64,activation='relu',input_shape=(n_features,)))
model_NN2.add(Dense(64,activation='relu'))
model_NN2.add(Dense(64,activation='relu'))
model_NN2.add(Dense(1))
model_NN2.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])
model_NN2.fit(X_train_transformed,y_train, batch_size=None, epochs=1,
              validation_split=0.0,validation_data=None)
test_performance_NN2 = model_NN1.evaluate(X_test_transformed,y_test)

# Graphing NN
viz_feature = 'displacement'
X_viz = data[[viz_feature]].values
y_viz = data[output].values
x_line = np.linspace(X_viz.min(), X_viz.max(), 440).reshape(-1, 1)
example_row = X.iloc[0].copy()
df_sweep = pd.DataFrame([example_row] * len(x_line))
df_sweep['displacement'] = x_line
X_sweep_transformed = preprocessor.transform(df_sweep)
y_line_nn1 = model_NN1.predict(X_train_transformed).flatten()
y_line_nn2 = model_NN2.predict(X_train_transformed).flatten()


plt.figure(figsize=(8, 5))
plt.scatter(X_viz,y_viz, alpha=0.4, label='Actual Data')
plt.plot(x_line, y_line_nn1, color='red', linewidth=2,
         label='NN1 Predicted Trend')
plt.plot(x_line, y_line_nn2, color='green', linewidth=2,
         label='NN2 Predicted Trend')

plt.title(f"Fuel Efficiency vs. {viz_feature.capitalize()}")
plt.xlabel(viz_feature.capitalize())
plt.ylabel("Combined MPG")
plt.legend()
plt.tight_layout()
plt.show()


# Categorical feature exploration
# for col in categorical:
#     print(f"\n{col}: {data[col].nunique()} unique classes")
#     print(data[col].value_counts().head(10))
# for col in categorical:
#     plt.figure(figsize=(8, 4))
#     data[col].value_counts().plot(kind='bar')
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Count")
# plt.show()