import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  #80% → Training 20% → Testing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
data = {
    "Time": [
        0, 5, 10, 15, 20,
        25, 30, 35, 40, 45,
        50, 55, 60, 65, 70,
        75, 80, 85, 90, 95
    ],

    "Temp": [
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
df=pd.DataFrame(data)
X = df[["Time","Temp","CmdPos"]] # input
y = df["Error"] #output
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)#basically x is the student who studies and y is using the answer key to check answers
model=LinearRegression()
model.fit(X_train,y_train) #basically means giving both question and answer for studying
prediction=model.predict(X_test)# no error column the model must guess
mae=mean_absolute_error(y_test,prediction) #actual answer vs predicted answer
print("Mean average error is ", mae)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
print(rmse)
r2 = r2_score(
    y_test,
    prediction
)

print(r2)
new_data = pd.DataFrame({
    "Time":[160],
    "Temp":[39],
    "CmdPos":[300]
})

predicted_error = model.predict(new_data)

print(predicted_error)
corrected_position = (
    300
    - predicted_error[0]
)
print(corrected_position)