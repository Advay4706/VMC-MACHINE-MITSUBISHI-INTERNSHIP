import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
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
        100,100,100,100,100,
        100,100,100,100,100,
        100,100,100,100,100
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
print(df.head())
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
y = df["Error"]
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
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

print("MAE =", mae)
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

print("RMSE =", rmse)
r2 = r2_score(
    y_test,
    prediction
)

print("R2 =", r2)
new_data = pd.DataFrame({
    "Time":[150],
    "SpindleRPM":[10000],
    "SpindleRuntime":[150],
    "StartStopCount":[15],
    "HeatIndex":[10000*150],
    "StartStopStress":[15*10000],
    "CmdPos":[100]
})
predicted_error = model.predict(
    new_data
)

print(predicted_error)
corrected_position = (
    100
    - predicted_error[0]
)

print(corrected_position)
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
print(
    mean_absolute_error(
        y_test,
        prediction2
    )
)

print(
    np.sqrt(
        mean_squared_error(
            y_test,
            prediction2
        )
    )
)

print(
    r2_score(
        y_test,
        prediction2
    )
)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model2.feature_importances_
})

print(
    importance.sort_values(
        by="Importance",
        ascending=False
    )
)
