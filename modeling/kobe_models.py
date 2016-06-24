import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.grid_search import GridSearchCV
%matplotlib inline
df = pd.read_csv('cleaned_kobe.csv')

# Split our data into a train and a testing set_cmap

train = df[df.shot_made_flag.isnull() == False]
test = df[df.shot_made_flag.isnull() == True]

# Pare down this list of features

X = train.drop(['shot_made_flag', 'action_type', 'combined_shot_type', 'season', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
                'team_name', 'game_date', 'matchup', 'opponent', 'days_off'], axis = 1)
y = train.shot_made_flag
kbest = SelectKBest(k=30)
kbest.fit(X,y)
# Show the feature importance for Kbest of 30
kbest_importance = pd.DataFrame(zip(X.columns, kbest.get_support()), columns = ['feature', 'important?'])

kbest_features = kbest_importance[kbest_importance['important?'] == True].feature
#Here's our dataframe
X_model = X[kbest_features]

x_train, x_test, y_train, y_test = train_test_split(X_model, y)
# Let the modelling begin

all_scores = {}
def evaluate_model(estimator, title):
    model = estimator.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred)
    con_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print "Accuracy Score:", acc_score.round(8)
    print
    print "Confusion Matrix:\n", con_matrix
    print
    print "Classification Report:\n", class_report
    all_scores[title] = acc_score
    print all_scores


# Models to test
lr = LogisticRegression(penalty = 'l1', C=0.2)
dt = DecisionTreeClassifier()
xt = ExtraTreesClassifier()
knn = KNeighborsClassifier()
svc = SVC()
rfc = RandomForestClassifier()
ab = AdaBoostClassifier(base_estimator = dt)


evaluate_model(lr, 'LogisticRegression')
evaluate_model(dt, 'Decision Tree')
evaluate_model(xt, 'Extra Trees')
evaluate_model(knn, 'KNeighbors')
evaluate_model(svc, 'Support Vector Classifier')
evaluate_model(rfc, 'Random Forest')
evaluate_model(ab, 'AdaBoost')
params ={"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2, 3, 4, 5, 6, 7]}
clf = GridSearchCV(ab, param_grid = params, cv = 3, scoring = 'roc_auc', n_jobs=-1, verbose = 1)
clf.fit(X_model, y)
clf.best_score_

lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

score = roc_auc_score(y_test, predictions)
score

test_predictions = pd.DataFrame(lr.predict_proba(test[kbest_features]), columns = ['miss', 'shot_made_flag'])
test_predictions.head()

df = pd.DataFrame(test['shot_id'])
df.reset_index(inplace=True, drop = True)
submission = df.join(test_predictions.shot_made_flag)
submission.set_index('shot_id',drop=True,inplace=True)

submission.head()

submission.to_csv('kobe_final_submission.csv')

# This subission has a log loss score of 0.61
