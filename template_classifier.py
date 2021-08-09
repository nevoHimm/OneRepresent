from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd

data = pd.read_csv('data_try.csv', encoding="ISO-8859-1")
data = pd.read_csv('data_try.csv', encoding="ISO-8859-1")

feature_names = data[data.columns[:-1]]
label_name = data[data.columns[-1]]

feat_train, feat_test, label_train, label_test = \
    train_test_split(feature_names, label_name, test_size=0.32, random_state=1)


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(feat_train, label_train)
test_pred = knn.predict(feat_test)
print(test_pred)
print("kNN model accuracy:", metrics.accuracy_score(label_test, test_pred))
# print("kNN model accuracy:", metrics.accuracy_score(label_test, test_pred))

