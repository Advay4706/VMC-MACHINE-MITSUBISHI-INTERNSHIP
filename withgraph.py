import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Sample Dataset

data = {
    "Time":[
        0,10,20,30,40,
        50,60,70,80,90,
        100,110,120,130,140
    ],

    "SpindleRPM":[
        0,2000,3000,4000,5000,
        5000,6000,6000,7000,7000,
        8000,8000,9000,9000,10000
    ],

    "SpindleRuntime":[
        0,10,20,30,40,
        50,60,70,80,90,
        100,110,120,130,140
    ],

    "StartStopCount":[
        0,1,2,3,4,
        5,5,6,7,8,
        9,10,11,12,13
    ],

    "CmdPos":[
        50,100,150,200,250,
        300,50,100,150,200,
        250,300,50,100,150
    ],

    "Error":[
        0.000,
        0.005,
        0.010,
        0.016,
        0.023,
        0.030,
        0.039,
        0.048,
        0.059,
        0.071,
        0.084,
        0.098,
        0.113,
        0.129,
        0.146
    ]
}

# Feature Engineering

df = pd.DataFrame(data)

df["HeatIndex"] = (
    df["SpindleRPM"]
    *
    df["SpindleRuntime"]
)

df["StartStopStress"] = (
    df["StartStopCount"]
    *
    df["SpindleRPM"]
)

print("\nDataset Preview\n")
print(df.head())

# Inputs

X = df[
    [
        "Time",
        "SpindleRPM",
        "SpindleRuntime",
        "StartStopCount",
        "HeatIndex",
        "StartStopStress",
        "CmdPos"
    ]
]

# Output

y = df["Error"]

# Train Test Split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# LINEAR REGRESSION
# =====================================

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

prediction = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

r2 = r2_score(
    y_test,
    prediction
)

print("MAE =", mae)
print("RMSE =", rmse)
print("R2 =", r2)

# Compensation Example

new_data = pd.DataFrame({

    "Time":[150],

    "SpindleRPM":[10000],

    "SpindleRuntime":[150],

    "StartStopCount":[15],

    "HeatIndex":[10000*150],

    "StartStopStress":[15*10000],

    "CmdPos":[300]

})

predicted_error = model.predict(
    new_data
)

print(
    "\nPredicted Error (Linear Regression) =",
    predicted_error[0]
)

corrected_position = (
    300
    -
    predicted_error[0]
)

print(
    "Compensated Position =",
    corrected_position
)

# =====================================
# RANDOM FOREST
# =====================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

model2 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model2.fit(
    X_train,
    y_train
)

prediction2 = model2.predict(
    X_test
)

mae2 = mean_absolute_error(
    y_test,
    prediction2
)

rmse2 = np.sqrt(
    mean_squared_error(
        y_test,
        prediction2
    )
)

r22 = r2_score(
    y_test,
    prediction2
)

print("MAE =", mae2)
print("RMSE =", rmse2)
print("R2 =", r22)

# Compensation Example

predicted_error2 = model2.predict(
    new_data
)

print(
    "\nPredicted Error (Random Forest) =",
    predicted_error2[0]
)

corrected_position2 = (
    300
    -
    predicted_error2[0]
)

print(
    "Compensated Position =",
    corrected_position2
)

# =====================================
# CROSS VALIDATION
# =====================================

scores = cross_val_score(
    model2,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\nCross Validation Scores")

print(scores)

print(
    "Average CV Score =",
    scores.mean()
)

# =====================================
# FEATURE IMPORTANCE
# =====================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model2.feature_importances_

})

importance_sorted = (
    importance.sort_values(
        by="Importance",
        ascending=False
    )
)

print("\nFeature Importance\n")

print(importance_sorted)

# =====================================
# RESULTS TABLE
# =====================================

results = pd.DataFrame({

    "ActualError": y_test,

    "PredictedError": prediction2

})

results["CompensatedPosition"] = (
    300
    -
    results["PredictedError"]
)

print("\nCompensation Table\n")

print(results)

# =====================================
# SAVE CSV
# =====================================

results.to_csv(
    "thermal_compensation_results.csv",
    index=False
)

print(
    "\nResults saved to thermal_compensation_results.csv"
)

# =====================================
# ACTUAL VS PREDICTED GRAPH
# =====================================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    y_test.values,
    marker="o",
    label="Actual Error"
)

plt.plot(
    prediction2,
    marker="x",
    label="Predicted Error"
)

plt.title(
    "Actual vs Predicted Error"
)

plt.xlabel(
    "Sample Number"
)

plt.ylabel(
    "Error (mm)"
)

plt.legend()

plt.grid(True)

plt.show()

# =====================================
# FEATURE IMPORTANCE GRAPH
# =====================================

importance_graph = (
    importance.sort_values(
        by="Importance",
        ascending=True
    )
)

plt.figure(
    figsize=(8,5)
)

plt.barh(
    importance_graph["Feature"],
    importance_graph["Importance"]
)

plt.title(
    "Feature Importance"
)

plt.xlabel(
    "Importance Score"
)

plt.show()