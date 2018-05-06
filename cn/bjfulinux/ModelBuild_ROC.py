import pandas as pd
from sklearn import svm, metrics
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from graphviz import Digraph
import pydot

from sklearn.naive_bayes import GaussianNB

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

# build GBM model
model_xgb = xgb.XGBClassifier(
    silent=False,
    n_estimators=500,
    max_depth=4,
    gamma=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    min_child_weight=2,
    n_jobs=4,
    scale_pos_weight=1).fit(X_train, y_train.values.ravel())

predictions_proba_xgb = model_xgb.predict_proba(X_test)


# Bayes model
model_bayes = GaussianNB()
model_bayes.fit(X_train, y_train.values.ravel())

predictions_proba_bayes = model_bayes.predict_proba(X_test)

# SVM model
model_svm = svm.SVC(kernel='rbf', probability=True)
model_svm.fit(X_train, y_train.values.ravel())

predictions_proba_svm = model_svm.predict_proba(X_test)


# ROC
fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test, predictions_proba_xgb[:,1])
fpr_bayes, tpr_bayes, thresholds = roc_curve(y_test, predictions_proba_bayes[:,1])
fpr_svm, tpr_svm, thresholds = roc_curve(y_test, predictions_proba_svm[:,1])

roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
roc_auc_bayes = auc(fpr_bayes, tpr_bayes)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure()
lw = 2
plt.plot(fpr_xgb, tpr_xgb, color='darkorange',
         lw=lw, label='Xgboost ROC curve (area = %0.2f)' % roc_auc_xgb)
plt.plot(fpr_bayes, tpr_bayes, color='darkgreen',
         lw=lw, label='Bayes ROC curve (area = %0.2f)' % roc_auc_bayes)
plt.plot(fpr_svm, tpr_svm, color='darkblue',
         lw=lw, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
