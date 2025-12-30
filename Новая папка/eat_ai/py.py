import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


data = pd.DataFrame({
"fruit": ["apple", "orange", "banana", "pineapple", "orange","apple"],
"weight": [100, 80, 50, 90, 125, 150],
"label": ["low", "low", "average", "low", "hight", "hight"]})





encoder = OneHotEncoder()
x_encoded = encoder.fit_transform(data[["fruit"]])

X = np.hstack([x_encoded.toarray(), data[["weight"]].values])
y = data["label"]
clf = DecisionTreeClassifier()
clf.fit(X,y)

new = encoder.transform([["banana"]]).toarray()
new_data = np.hstack((new, [[125]]))
print(clf.predict(new_data))
    













