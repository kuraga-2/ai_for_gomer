from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
sales = [100, 200, 150, 120]
def moving_average(series, window=3):

    if len(series) < window:
        return np.mean(series)
    return np.mean(series[-window])
ma_pred = moving_average(sales, window=3)
print(f'moving average prediction to next week: {ma_pred}')





x = np.array(len(sales)).reshape(-1,1)
y = sales
model = LinearRegression()
model.fit(x,y)
next_week = np.array([[len(sales)]])
linear_pred = model.predict(next_week)[0]
print(f"linear regression prediction next week: {linear_pred}")