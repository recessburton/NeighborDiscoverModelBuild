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

"""
# build svm model
model = svm.SVC(kernel='rbf')
model.fit(x_train, y_train.values.ravel())"""

# build GBM model
model = xgb.XGBClassifier(
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

"""
if __name__ == '__main__':
    # Choose some parameter combinations to try
    parameters = {'n_estimators': [400],
                  'max_depth': [4],
                  'gamma': [0.8]
                  }
    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)
    grid_search = GridSearchCV(estimator= model, param_grid=parameters, scoring=acc_scorer,
                           n_jobs=4, cv=2, refit=True, verbose=2)
    grid_search.fit(X_train, y_train)
    # Set the clf to the best combination of parameters
    clf = grid_search.best_estimator_
    # Fit the best algorithm to the data.
    clf.fit(X_train, y_train)
"""

predictions = model.predict(X_test)

# save model
#pickle.dump(model, open('xgboost.model', 'wb'))


cm_train = metrics.confusion_matrix(y_train, model.predict(X_train))
cm_test = metrics.confusion_matrix(y_test, model.predict(X_test))

print(cm_train)
print(model.score(X_train, y_train))
print(cm_test)
print(model.score(X_test, y_test))

predictions_proba = model.predict_proba(X_test)

print("XGBoost_自带接口    AUC Score : %f" % metrics.roc_auc_score(y_test, predictions))


xgb.plot_tree(model, num_trees=4)
xgb.plot_importance(model)
plt.show()

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


xgb.plot_tree(model)
exit(0)


#################

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
ceate_feature_map(features)

xgb.plot_tree(model,fmap='xgb.fmap')

import re

_NODEPAT = re.compile(r'(\d+):\[(.+)\]')
_LEAFPAT = re.compile(r'(\d+):(leaf=.+)')
_EDGEPAT = re.compile(r'yes=(\d+),no=(\d+),missing=(\d+)')
_EDGEPAT2 = re.compile(r'yes=(\d+),no=(\d+)')


def _parse_node(graph, text):
    """parse dumped node"""
    match = _NODEPAT.match(text)
    if match is not None:
        node = match.group(1)
        graph.node(node, label=match.group(2), shape='plaintext')
        return node
    match = _LEAFPAT.match(text)
    if match is not None:
        node = match.group(1)
        graph.node(node, label=match.group(2).replace('leaf=',''), shape='plaintext')
        return node
    raise ValueError('Unable to parse node: {0}'.format(text))


def _parse_edge(graph, node, text, yes_color='#0000FF', no_color='#FF0000'):
    """parse dumped edge"""
    try:
        match = _EDGEPAT.match(text)
        if match is not None:
            yes, no, missing = match.groups()
            if yes == missing:
                graph.edge(node, yes, label='yes, missing', color=yes_color)
                graph.edge(node, no, label='no', color=no_color)
            else:
                graph.edge(node, yes, label='yes', color=yes_color)
                graph.edge(node, no, label='no, missing', color=no_color)
            return
    except ValueError:
        pass
    match = _EDGEPAT2.match(text)
    if match is not None:
        yes, no = match.groups()
        graph.edge(node, yes, label='yes', color=yes_color)
        graph.edge(node, no, label='no', color=no_color)
        return
    raise ValueError('Unable to parse edge: {0}'.format(text))


from graphviz import Digraph
booster = model.get_booster()
tree = booster.get_dump(fmap='xgb.fmap')[0]
tree = tree.split()


kwargs = {
        #'label': 'A Fancy Graph',
        'fontsize': '10',
        #'fontcolor': 'white',
        #'bgcolor': '#333333',
        #'rankdir': 'BT'
         }
kwargs = kwargs.copy()
#kwargs.update({'rankdir': rankdir})
graph = Digraph(format='pdf', node_attr=kwargs,edge_attr=kwargs,engine='dot')#,edge_attr=kwargs,graph_attr=kwargs,
#graph.attr(bgcolor='purple:pink', label='agraph', fontcolor='white')

yes_color='#0000FF'
no_color='#FF0000'
for i, text in enumerate(tree):
    if text[0].isdigit():
        node = _parse_node(graph, text)
    else:
        if i == 0:
            # 1st string must be node
            raise ValueError('Unable to parse given string as tree')
        _parse_edge(graph, node, text, yes_color=yes_color,no_color=no_color)


graph.render('XGBoost_tree.pdf')

graph
#################



