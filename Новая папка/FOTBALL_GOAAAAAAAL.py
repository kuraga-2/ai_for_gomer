from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
goals = [2,3,4,5,6,7]
def moving_average(series, window=6):

    if len(series) < window:
        return np.mean(series)
    return np.mean(series[-window])
ma_pred = moving_average(goals, window=6)
print(f'moving average prediction to next week: {ma_pred}')



x = np.arange(len(goals)).reshape(-1, 1)
print(x)
y = goals
model = LinearRegression()
model.fit(x,y)
next_week = np.array([[len(goals)]])
linear_pred = model.predict(next_week)[0]
print(f"linear regression prediction next week: {linear_pred}")