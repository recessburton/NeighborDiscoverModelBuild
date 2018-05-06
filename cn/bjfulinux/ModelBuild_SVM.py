import pandas as pd
from sklearn import svm, metrics
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from graphviz import Digraph
import pydot

input_data = pd.read_table("neighbor-log.txt", sep=' ', header=None,
                           names=['distance_r_p', 'distance_r_t', 'distance_p_t', 'rssi_r_p', 'rssi_r_t', 'com_r_p',
                                  'com_r_t', 'com_p_t', 'com_r_p_t', 'max_rssi', 'avg_rssi', 'noise_thre', 'class'])

df = pd.DataFrame(input_data)

features = ['distance_r_p', 'distance_r_t', 'distance_p_t', 'rssi_r_p', 'rssi_r_t', 'com_r_p', 'com_r_t', 'com_p_t',
            'com_r_p_t', 'max_rssi', 'avg_rssi', 'noise_thre']
#features = ['distance_r_p', 'distance_r_t', 'distance_p_t', 'rssi_r_p', 'rssi_r_t', 'com_r_p', 'com_r_t', 'com_p_t',
#            'com_r_p_t']


X_all = df.loc[:, features]
y_all = df['class']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

"""
# data standardization
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)"""


# build svm model
model = svm.SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train.values.ravel())

predictions = model.predict(X_test)

# save model
# pickle.dump(model, open('svm.model', 'wb'))



cm_train = metrics.confusion_matrix(y_train, model.predict(X_train))
cm_test = metrics.confusion_matrix(y_test, model.predict(X_test))

print(cm_train)
print(model.score(X_train, y_train))
print(cm_test)
print(model.score(X_test, y_test))

print("AUC Score : %f" % metrics.roc_auc_score(y_test, predictions))

predictions_proba = model.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, predictions_proba[:,1])

roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()




