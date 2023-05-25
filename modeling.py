import re
import pandas as pd
import numpy as np
import plotly.express as px

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score , recall_score , precision_score , f1_score , roc_curve , auc, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

RANDOM_STATE = 1823

AAC_TR = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/AAC/AAC-TR.csv")
AAC_TS = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/AAC/AAC-TS.csv")

APAAC_TR = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/APAAC/APAAC-TR.csv")
APAAC_TS = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/APAAC/APAAC-TS.csv")

PAAC_TR = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/PAAC/PAAC-TR.csv")
PAAC_TS = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/PAAC/PAAC-TS.csv")

RSacid_TR = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/RSacid/RSacid-TR.csv")
RSacid_TS = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/RSacid/RSacid-TS.csv")

RScharge_TR = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/RScharge/RScharge-TR.csv")
RScharge_TS = pd.read_csv("/content/drive/MyDrive/Predicting druggable proteins Assignment/output/RScharge/RScharge-TS.csv")

APAAC_TR.shape

"""## Q2"""

datasets = {  
    "data": [
      {
          "train": AAC_TR,
          "test": AAC_TS,
          "target" : "id",
          "cols" : ['AAC_0', 'AAC_1', 'AAC_2', 'AAC_3', 'AAC_4', 'AAC_5',
        'AAC_6', 'AAC_7', 'AAC_8', 'AAC_9', 'AAC_10', 'AAC_11', 'AAC_12',
        'AAC_13', 'AAC_14', 'AAC_15', 'AAC_16', 'AAC_17', 'AAC_18', 'AAC_19']
      },
      {
        "train": APAAC_TR,
        "test": APAAC_TS,
        "target" : "id",
        "cols" : ['APAAC_0', 'APAAC_1', 'APAAC_2', 'APAAC_3', 'APAAC_4',
       'APAAC_5', 'APAAC_6', 'APAAC_7', 'APAAC_8', 'APAAC_9', 'APAAC_10',
       'APAAC_11', 'APAAC_12', 'APAAC_13', 'APAAC_14', 'APAAC_15', 'APAAC_16',
       'APAAC_17', 'APAAC_18', 'APAAC_19', 'APAAC_20', 'APAAC_21']
      },
      {
        "train": PAAC_TR,
        "test": PAAC_TS,
        "target" : "id",
        "cols" : ['PAAC_0', 'PAAC_1', 'PAAC_2', 'PAAC_3', 'PAAC_4',
       'PAAC_5', 'PAAC_6', 'PAAC_7', 'PAAC_8', 'PAAC_9', 'PAAC_10', 'PAAC_11',
       'PAAC_12', 'PAAC_13', 'PAAC_14', 'PAAC_15', 'PAAC_16', 'PAAC_17',
       'PAAC_18', 'PAAC_19', 'PAAC_20']
      },
      {
        "train": RSacid_TR,
        "test": RSacid_TS,
        "target" : "id",
        "cols" : ['RSacid_0', 'RSacid_1', 'RSacid_2', 'RSacid_3',
       'RSacid_4', 'RSacid_5', 'RSacid_6', 'RSacid_7', 'RSacid_8', 'RSacid_9',
       'RSacid_10', 'RSacid_11', 'RSacid_12', 'RSacid_13', 'RSacid_14',
       'RSacid_15', 'RSacid_16', 'RSacid_17', 'RSacid_18', 'RSacid_19',
       'RSacid_20', 'RSacid_21', 'RSacid_22', 'RSacid_23', 'RSacid_24',
       'RSacid_25', 'RSacid_26', 'RSacid_27', 'RSacid_28', 'RSacid_29',
       'RSacid_30', 'RSacid_31']
      },
      {
        "train": RScharge_TR,
        "test": RScharge_TS,
        "target" : "id",
        "cols" : ['RScharge_0', 'RScharge_1', 'RScharge_2',
       'RScharge_3', 'RScharge_4', 'RScharge_5', 'RScharge_6', 'RScharge_7',
       'RScharge_8', 'RScharge_9', 'RScharge_10', 'RScharge_11', 'RScharge_12',
       'RScharge_13', 'RScharge_14', 'RScharge_15', 'RScharge_16',
       'RScharge_17', 'RScharge_18', 'RScharge_19', 'RScharge_20',
       'RScharge_21', 'RScharge_22', 'RScharge_23', 'RScharge_24',
       'RScharge_25', 'RScharge_26', 'RScharge_27', 'RScharge_28',
       'RScharge_29', 'RScharge_30', 'RScharge_31']
      }
    ]

}

models = [
    {
        "name": "XGBoost",
        "clf": XGBClassifier()
    },
    {
        "name": "LGBMClassifier",
        "clf": LGBMClassifier()
    },
    {
        "name": "DecisionTree",
        "clf": DecisionTreeClassifier()
    },
    {
        "name": "SVM",
        "clf": SVC()
    },
    {
        "name": "RandomForest",
        "clf": RandomForestClassifier()
    },
    {
        "name": "KNN",
        "clf": KNeighborsClassifier(n_neighbors=3)
    }
]

import inspect
def encode(x):
  if "Positive" in x:
    return 1
  else:
    return 0

for dataset in datasets["data"]:
  for model in models:
  
    x_train = dataset["train"][dataset["cols"]]
    x_test = dataset["test"][dataset["cols"]]
    y_train = dataset["train"]["id"].apply(encode)
    y_test = dataset["test"]["id"].apply(encode)

    # print(f"==========================={dataset['cols'][0].split('_')[0]} - {model['name']}==========================")
    clf = model["clf"]
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Sensitivity:", sensitivity)
    # print("Specificity:", specificity)
    # print("Precision:", precision_score(y_test, y_pred))
    # print("F1 Score:", f1_score(y_test, y_pred))
    hyperparams = inspect.signature(clf.__init__)
    print(hyperparams)



"""## Q3"""

x_train = pd.concat([ dataset["train"][dataset["cols"]] for dataset in datasets["data"]], axis = 1)
x_test = pd.concat([ dataset["test"][dataset["cols"]] for dataset in datasets["data"]], axis = 1)

y_train = datasets["data"][0]["train"]["id"].apply(encode)
y_test = datasets["data"][0]["test"]["id"].apply(encode)

x_train.columns

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = XGBClassifier()
model.fit(x_train,y_train)

feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
# feat_importances.nlargest(40).plot(kind='barh')
# plt.show()
# feat_importances.sort_values()

for n in [10,20,30]:
  cols = feat_importances.sort_values(ascending=[0]).reset_index()['index'][:n]
  _x_train = x_train[cols]
  _x_test = x_test[cols]
  xgb = XGBClassifier()
  xgb.fit(_x_train,y_train)
  y_pred = xgb.predict(_x_test)
  print(f"F1 Score for {n} features:", f1_score(y_test, y_pred))

selected_features = feat_importances.sort_values(ascending=[0]).reset_index()['index'][:20]

x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

selected_features

for model in models:
  # print(cross_val_score(clf, x_train, y_train, cv=5))
  print(model["name"], np.mean(cross_val_score(model["clf"], x_train_selected, y_train, cv=5)))

def grid_search(params, random=False):

    XGB_CLF = LGBMClassifier( booster='gbtree', random_state=1823, nthread=-1,)
   
    grid = GridSearchCV(estimator=XGB_CLF, param_grid=params, n_jobs=-1 , cv=5, return_train_score=True)
    if random: grid = RandomizedSearchCV(estimator=XGB_CLF, param_distributions=params, n_iter=100, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)

    grid.fit(x_train_selected.to_numpy(), y_train.to_numpy())

    return grid

params={100,
        max_depth5,

        }

grid = grid_search(params,random=True)

grid.best_params_

x_train , x_test , y_train , y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=1823)

xgb_clf = XGBClassifier()
xgb_clf.fit(x_train, y_train)
y_pred = xgb_clf.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

random_forest = RandomForestClassifier(n_estimators=100)

cross_val_preds = cross_val_predict(random_forest , X_train,Y_train)
confusion_matrix(Y_train,cross_val_preds)

print(cross_val_score(random_forest, X_train, Y_train, cv=5))
print(np.mean(cross_val_score(random_forest, X_train, Y_train, cv=5)))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))