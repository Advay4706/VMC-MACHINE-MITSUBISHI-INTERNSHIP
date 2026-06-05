import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #80% → Training 20% → Testing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
data = {
    "Time": [
        0, 5, 10, 15, 20,
        25, 30, 35, 40, 45,
        50, 55, 60, 65, 70,
        75, 80, 85, 90, 95
    ],
    "Temp": 
    [
        22.00, 23.73, 25.29, 26.68, 27.89,
        28.92, 29.81, 30.56, 31.23, 31.83,
        32.41, 32.98, 33.57, 34.17, 34.78,
        35.38, 35.95, 36.45, 36.87, 37.20
    ],

    "CmdPos": [
        50, 150, 300, 50, 150,
        300, 50, 150, 300, 50,
        150, 300, 50, 150, 300,
        50, 150, 300, 50, 150
    ],

    "Error": [
        0.0010, 13.1572, 25.0238, 35.5929, 44.7965,
        52.6269, 59.3927, 65.0983, 70.1915, 74.7576,
        79.1684, 83.5031, 87.9929, 92.5543, 97.1946,
        101.7553, 106.0877, 109.8920, 113.0839, 115.5960
    ]
}

df = pd.DataFrame(data)
# thermal expansion constants
alpha = 11.7e-6
length = 650
reference_temp = 22
# temperature rise above starting temperature
df["TempRise"] = df["Temp"] - reference_temp
# theoretical expansion using ΔL = α × L × ΔT
df["TheoExp"] = (
    alpha *
    length *
    df["TempRise"]
)
X = df[
    [
        "Time",
        "Temp",
        "CmdPos",
        "TempRise",
        "TheoExp"
    ]
] # input
y = df["Error"] # output
# basically x is the student who studies and y is using the answer key to check answers
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model=LinearRegression()
# basically means giving both question and answer for studying
model.fit(X_train,y_train)
# no error column the model must guess
prediction=model.predict(X_test)
# actual answer vs predicted answer
mae=mean_absolute_error(y_test,prediction)
print("Mean average error is ", mae)
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)
print(rmse)
r2 = r2_score(
    y_test,
    prediction
)
print(r2)
temp_rise = 39 - reference_temp
theo_exp = (
    alpha *
    length *
    temp_rise
)
new_data = pd.DataFrame({
    "Time":[160],
    "Temp":[39],
    "CmdPos":[300],
    "TempRise":[temp_rise],
    "TheoExp":[theo_exp]
})
predicted_error = model.predict(new_data)
print(predicted_error)
corrected_position = (
    300
    - predicted_error[0]
)
print(corrected_position)
# NOW RANDOM FOREST
model2=RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model2.fit(X_train,y_train)
prediction2=model2.predict(X_test)
mae2 = mean_absolute_error(
    y_test,
    prediction2
)
print(mae2)
rmse2 = np.sqrt(
    mean_squared_error(
        y_test,
        prediction2
    )
)
print(rmse2)
r22 = r2_score(
    y_test,
    prediction2
)
print(r22)
temp_rise2 = 39 - reference_temp
theo_exp2 = (
    alpha *
    length *
    temp_rise2
)
new_data2 = pd.DataFrame({
    "Time":[160],
    "Temp":[39],
    "CmdPos":[300],
    "TempRise":[temp_rise2],
    "TheoExp":[theo_exp2]
})
predicted_error2 = model2.predict(
    new_data2
)
print(predicted_error2)
corrected_position2 = (
    300 -
    predicted_error2[0]
)
print(corrected_position2)
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