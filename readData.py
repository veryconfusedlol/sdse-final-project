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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# ====== PREPARE NN DATA  ======

# 1. Fit preprocessor ONLY on training data
preprocessor.fit(X_train)

# 2. Transform train and test sets
X_train_enc = preprocessor.transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

# 3. Scale features (sparse safe)
scaler = StandardScaler(with_mean=False)
scaler.fit(X_train_enc)

X_train_enc = scaler.transform(X_train_enc)
X_test_enc  = scaler.transform(X_test_enc)

# 4. Convert to dense matrices for Keras
X_train_enc = X_train_enc.toarray()
X_test_enc  = X_test_enc.toarray()

print("Final input feature dimension:", X_train_enc.shape[1])

# ====== NN MODEL  ======
# Model A - Simple 1-hidden-layer network
# Architecture: Input → Dense(32, ReLU) → Dense(1)


def build_model_A(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Model B — Deeper network
# Architecture: Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1)

def build_model_B(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Model C — Add dropout (regularized)
# Architecture: Input → Dense(64, ReLU) → Dropout(0.25) → Dense(32, ReLU) → Dense(1)
def build_model_C(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Model D — Wide network with batch normalization
# Architecture: Input → Dense(128, ReLU) → BatchNormalization → Dense(64, ReLU) → BatchNormalization → Dense(1)
from tensorflow.keras.layers import BatchNormalization

def build_model_D(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train and evaluate each model
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True)
]

models = {
    "Model A": build_model_A(X_train_enc.shape[1]),
    "Model B": build_model_B(X_train_enc.shape[1]),
    "Model C": build_model_C(X_train_enc.shape[1]),
    "Model D": build_model_D(X_train_enc.shape[1]),
}

history = {}
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    history[name] = model.fit(
        X_train_enc, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    test_mse, test_mae = model.evaluate(X_test_enc, y_test, verbose=0)
    results[name] = (test_mse, test_mae)
    print(f"{name} — Test MAE: {test_mae:.3f}")

# Plot training curves
for name in history:
    plt.figure(figsize=(6,4))
    plt.plot(history[name].history['loss'], label='Train Loss')
    plt.plot(history[name].history['val_loss'], label='Val Loss')
    plt.title(name)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()