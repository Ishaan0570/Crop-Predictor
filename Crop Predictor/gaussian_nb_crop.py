import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np

inputs = pd.read_csv('Crop_recommendation.csv')

le_label = LabelEncoder()

inputs['label_num'] = le_label.fit_transform(inputs["label"])

# To check the labels
# print(inputs['label'].value_counts())
inputs_n = inputs.drop(["N","P","K","label","label_num"],axis = "columns")
target = inputs["label_num"]
X_train, X_test, y_train, y_test = train_test_split(inputs_n,target,test_size=0.25)

model = GaussianNB()

model.fit(X_train,y_train)

predicted_values = model.predict(X_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("Naive Bayes's Accuracy is: ", x)

# # To check the important features in the algorithm
# importances = model.feature_importances_

# indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(16,8))
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), importances[indices])
# plt.xticks(range(X_train.shape[1]),X_train.columns[indices])
# plt.show()

# # To predict our values
# ans = model.predict([[25,69.57,67,100]])
# print(le_label.inverse_transform(ans)[-1])