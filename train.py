import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def compute_accuracy(Y_true, Y_pred):
    correctly_predicted = 0
    # iterating over every label and checking it with the true sample
    for true_label, predicted in zip(Y_true, Y_pred):
        if true_label == predicted:
            correctly_predicted += 1
    # computing the accuracy score
    accuracy_score = correctly_predicted / len(Y_true)
    return accuracy_score

df = pd.read_csv('dataset.csv')
df.columns = [i for i in range(df.shape[1])]
print(df)

df = df.rename(columns={63: 'Output'})
print(df)

X = df.iloc[:, :-1]
print("Features shape =", X.shape)

Y = df.iloc[:, -1]
print("Labels shape =", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
svm = SVC(C=10, gamma=0.1, kernel='rbf') #Support Vector Classifier
svm.fit(x_train, y_train) #Support Vector Machine

y_pred = svm.predict(x_test)
print(y_pred)

cf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
score = compute_accuracy(y_test, y_pred)

print('F1: ', f1)
print('Recall: ', recall)
print('Precision: ', precision)
print('Accuracy Score:', score)

labels = sorted(list(set(df['Output'])))
labels = [x.upper() for x in labels]

fig, ax = plt.subplots(figsize=(12, 12))

ax.set_title("Confusion Matrix - Sign Language")

maping = sns.heatmap(cf_matrix,
                     annot=True,
                     cmap = plt.cm.Blues,
                     linewidths=.2,
                     xticklabels=labels,
                     yticklabels=labels, vmax=8,
                     fmt='g',
                     ax=ax
                    )
# print(maping)
plt.show()

import pickle

# save model
with open('model.pkl','wb') as f:
    pickle.dump(svm,f)