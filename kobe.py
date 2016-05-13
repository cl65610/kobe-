# OUr goal here is to build out a dataframe that will be useful for modelling.

import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
%matplotlib inline

kobe = pd.read_csv('kobe.csv')

kobe.columns
kobe2 = kobe[kobe.shot_made_flag >= 0]

kobe2.shot_made_flag
kobe2.shape
kobe.head(30)
kobe2.shot_type.value_counts() # This only includes 2s and 3s, which might not be the most useful thing, especially if we can get it by creating dummies for
#another variable.

# What do the minutes_remaining and secons_remaining describe. Is seconds shot clock?

#This describes the minutes remaining in the quarter
kobe2.minutes_remaining.value_counts()

kobe2.seconds_remaining.value_counts()
# This is seconds remaining in the quarter, not on the shot clock. It probably won't be the most useful thing in the world.
# Get dummy variables for season
season_dummies = pd.get_dummies(kobe2.season)
kobe2['home_or_away'] = 

#join the dummies dataframe onto the original
kobe2 = kobe2.join(season_dummies)


kobe2.game_id.value_counts().count()
kobe2.game_date.nunique()

kobe2.dtypes
kobe2.shot_distance.value_counts()
kobe2.season.value_counts()
print kobe2[kobe2.game_date == '1996-11-03'].shot_made_flag
import seaborn as sns
plt.style.use('ggplot')
plt.set_cmap('seismic')
plt.figure(figsize=(24,22))
plt.scatter(kobe2.loc_x, kobe2.loc_y, alpha = 0.3, c=kobe2.shot_made_flag) # Charts all of Kobe's career shots by location and whether they were m akes or misses
plt.xlim(-300,300)
plt.ylim(-100,500)
# plt.savefig('Kobe_shot_chart.png')
plt.show()

kobe_plot = sns.pairplot(kobe2, hue='shot_made_flag', vars = ['loc_x', 'loc_y', 'period', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'playoffs'])
kobe_plot.savefig('kobe_pair_plot.png')


# Testing out some groupby functions

games = kobe2.groupby('game_date')

# Does the mean here give you average field goal percentage for that game?
games = games.agg({'shot_made_flag': [np.sum, np.mean],
                    'opponent': lambda x: x.iloc[0]})
kobe2.game_event_id.nunique()

print kobe2.columns
kobe2.combined_shot_type.value_counts()

feature_cols = ['loc_x', 'loc_y', 'minutes_remaining', 'seconds_remaining', 'shot_distance', '1996-97',
                '1997-98', '1998-99', '1999-00', '2000-01', '2001-02', '2002-03',
                '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09',
                '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
                '2015-16', 'playoffs']

# Let the modelling begin
X = kobe2[feature_cols]
y = kobe2.shot_made_flag
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

model = lr.fit(X_train, y_train)
y_class_pred = lr.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, y_class_pred)
lr.score(X_test, y_test)

plt.scatter(kobe2.shot_distance, kobe2.loc_y)

# Try a KNN model - It performs worse with these features

from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,66)
training_error = []
testing_error = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
#Training accuracy
    model = knn.fit(X_train, y_train)
    y_pred_class = knn.predict(X_train)
    train_accuracy = metrics.accuracy_score(y_train, y_pred_class)
    training_error.append(1-train_accuracy)

#Testing accuracy
    y_pred_class = knn.predict(X_test)
    testing_accuracy = metrics.accuracy_score(y_test, y_pred_class)
    testing_error.append(1-testing_accuracy)

column_dict = {'K': k_range, 'training error':training_error, 'testing error':testing_error}
df = pd.DataFrame(column_dict).set_index('K').sort_index(ascending=False)
df.plot(y='testing error')

metrics.accuracy_score(y_test, knn_predicts)
df.sort('testing error').head()


# Try it out with some different features

feature_cols = ['loc_x', 'loc_y', 'seconds_remaining', 'shot_distance', '1996-97',
                '1997-98', '1998-99', '1999-00', '2000-01', '2001-02', '2002-03',
                '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09',
                '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
                '2015-16', 'playoffs', 'game_event_id']

# Let the modelling begin
X = kobe2[feature_cols]
y = kobe2.shot_made_flag
lr = LogisticRegression()

model = lr.fit(X_train, y_train)
y_class_pred = lr.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, y_class_pred)
lr.score(X_test, y_test)

# After looking for some more useful variables. I think it might be useful to create dummies
# for the 'combined shot type' and see if that helps.

shot_dummies = pd.get_dummies(kobe2.combined_shot_type)

kobe2 = kobe2.join(shot_dummies)

kobe2.columns

feature_cols = ['loc_x', 'loc_y', 'seconds_remaining', 'shot_distance', '1996-97',
                '1997-98', '1998-99', '1999-00', '2000-01', '2001-02', '2002-03',
                '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09',
                '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
                '2015-16', 'playoffs', 'Bank Shot', 'Dunk', 'Hook Shot',
                'Jump Shot', 'Layup', 'Tip Shot', 'minutes_remaining']

X = kobe2[feature_cols]
y = kobe2.shot_made_flag
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

model = lr.fit(X_train, y_train)
y_class_pred = lr.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, y_class_pred)
lr.score(X_test, y_test)
