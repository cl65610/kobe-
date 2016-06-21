# OUr goal here is to build out a dataframe that will be useful for modelling.

import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
%matplotlib inline

kobe = pd.read_csv('kobe.csv')
kobe.shot_made_flag.head(25)
kobe.columns
kobe2 = kobe[kobe.shot_made_flag >= 0]

kobe_test = kobe[kobe.shot_made_flag.isnull() == True]
# What do the minutes_remaining and secons_remaining describe. Is seconds shot clock? It's not
# They're both the time left in the quarter.

#This describes the minutes remaining in the quarter
kobe2.minutes_remaining.value_counts()

kobe2.seconds_remaining.value_counts()
# This is seconds remaining in the quarter, not on the shot clock. It probably won't be the most useful thing in the world.
# Get dummy variables for season
season_dummies = pd.get_dummies(kobe2.season, drop_first=True)
#join the dummies dataframe onto the original
kobe2 = kobe2.join(season_dummies)

test_dummies = pd.get_dummies(kobe_test.season, drop_first=True)
kobe_test = kobe_test.join(test_dummies)

# Make a column that shows whether or not a game was at home
kobe2.iloc[0].matchup

kobe2['home'] = np.where(kobe2['matchup'].str.startswith('LAL @'), 0, 1) # matches the string and fills with a zero where True, and a one where False
kobe_test['home'] = np.where(kobe_test['matchup'].str.startswith('LAL @'), 0, 1) # matches the string and fills with a zero where True, and a one where False
# Create a column that combines minutes and seconds remaining.
kobe_test['time_left'] = kobe_test['minutes_remaining']+(kobe_test.seconds_remaining/60.0)

# Convert the shot_type column to have integer values for the type of shot.
kobe2.shot_type = kobe2.shot_type.map({'2PT Field Goal':2, '3PT Field Goal':3})
kobe_test.shot_type = kobe_test.shot_type.map({'2PT Field Goal':2, '3PT Field Goal':3})

# Based on the below, it looks like game id and game date could both be used for indexing individual games.
kobe2.game_id.value_counts().count()
kobe2.game_date.nunique()

# kobe2.dtypes
# kobe2.shot_distance.value_counts()
# kobe2.season.value_counts()
# print kobe2[kobe2.game_date == '1996-11-03'].shot_made_flag
# import seaborn as sns
# plt.style.use('ggplot')
# plt.set_cmap('seismic')
# plt.figure(figsize=(24,22))
# plt.scatter(kobe2.loc_x, kobe2.loc_y, alpha = 0.3, c=kobe2.shot_made_flag) # Charts all of Kobe's career shots by location and whether they were m akes or misses
# plt.xlim(-300,300)
# plt.ylim(-100,500)
# # plt.savefig('Kobe_shot_chart.png')
# plt.show()
#
# kobe_plot = sns.pairplot(kobe2, hue='shot_made_flag', vars = ['loc_x', 'loc_y', 'period', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'playoffs'])
# kobe_plot.savefig('kobe_pair_plot.png')

# After looking for some more useful variables. I think it might be useful to create dummies
# for the 'combined shot type' and see if that helps.
action_dummies = pd.get_dummies(kobe2.action_type, drop_first=True)
action_cols = action_dummies.columns
kobe2 = kobe2.join(action_dummies)
test_action = pd.get_dummies(kobe_test.action_type, drop_first=True)
test_cols = test_action.columns
kobe_test = kobe_test.join(test_action)
# Create a column for days off
# The first thing we're going to have to do is translate the game date from an object to a date

kobe2.game_date = kobe2.game_date.astype('datetime64')
kobe_test.game_date = kobe_test.game_date.astype('datetime64')
# Now, we should be able to create this new calculated column using a groupby

games = kobe2.groupby('game_id', as_index=False)

# Does the mean here give you average field goal percentage for that game?

# In order to get this number, we're going to have to calculate it in a grouped thing,
# convert those results to a dictionary, and then map that dictionary over the original df.
games = games.agg({'shot_made_flag': [np.sum, np.mean],
                    'opponent': lambda x: x.iloc[0],
                    'game_date':lambda x:x.iloc[0]})
games.columns = ['_'.join(col).strip() for col in games.columns.values]
games.columns
games['days_off'] = games['game_date_<lambda>'] - games['game_date_<lambda>'].shift() #With this calculation, 1 day off
# means that it was a b2b.
games.days_off.sort_values()
# We need some # of days that will be a cutoff. Maybe 7 days?
days_off_dict = {}
days_off_dict['games.game_date_<lambda>'] = games.days_off

days_off_dict
a=[]
for row in games.game_id_:
    a.append(row)
b=[]
for row in games.days_off:
    b.append(row)

print b

days_off_dict = dict(zip(a,b))
days_off_dict

kobe2.game_id.head()
kobe2['days_off'] = kobe2.game_id
kobe2['days_off'] = kobe2.game_id.map(days_off_dict)
kobe2.days_off.fillna(100, inplace=True)
kobe2.days_off = kobe2.days_off.astype('timedelta64[D]')
kobe2.days_off = kobe2.days_off.apply(lambda x: x-1)
kobe2.days_off = kobe2.days_off.apply(lambda x: x==10 if x>=11 else x)
kobe2.days_off = kobe2.days_off.apply(lambda x: x+11 if x<0 else x)
kobe2.days_off.max()
kobe2.days_off.head(15)



print kobe2.columns
kobe2.combined_shot_type.value_counts()

feature_cols = ['loc_x', 'loc_y', 'minutes_remaining', 'seconds_remaining', 'shot_distance',
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
df['testing error'].max()

metrics.accuracy_score(y_test, knn_predicts)
df.sort('testing error').head()

# This run of KNN doesn't get close to logistic regression


# Try it out with some different features

feature_cols = ['loc_x', 'loc_y', 'seconds_remaining', 'shot_distance',
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


kobe2.action_type.value_counts()
kobe2.columns
kobe2.action_type.value_counts()


kobe2.shape
period_dummies = pd.get_dummies(kobe2.period, drop_first=True)

kobe2 =kobe2.join(period_dummies)

# Calculate how much time has elapsed in the game
action_cols
feature_cols = ['loc_x', 'loc_y', 'Alley Oop Layup shot',
                 'Driving Dunk Shot', 'Driving Finger Roll Layup Shot',
                'Driving Finger Roll Shot',
                 'Driving Layup Shot', 'Driving Reverse Layup Shot',
                'Driving Slam Dunk Shot', 'Dunk Shot', 'Fadeaway Bank shot', 'Fadeaway Jump Shot', 'Finger Roll Layup Shot',
                'Finger Roll Shot', 'Floating Jump shot',
                'Hook Shot', 'Jump Bank Shot', 'Jump Shot', 'Layup Shot', 'Pullup Jump shot',
                 'Running Bank shot',
                'Running Finger Roll Layup Shot', 'Running Hook Shot', 'Running Jump Shot', 'Running Layup Shot',
                'Slam Dunk Shot', 'Step Back Jump shot', 'Tip Shot', 'Turnaround Bank shot', 'Turnaround Fadeaway shot',
                 'Turnaround Jump Shot']

X = kobe2[feature_cols]
X.set_index()
y = kobe2.shot_made_flag
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.7)

model = lr.fit(X_train, y_train)
y_class_pred = lr.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, y_class_pred)
lr.score(X_test, y_test)


shot_predictions = lr.predict(kobe_test[feature_cols])


made_predictions = pd.DataFrame(shot_predictions, columns = ['shot_made_flag'])
submission = pd.DataFrame(kobe_test.shot_id)
submission.head()

submission = submission.join(made_predictions)
submission = submission.set_index('shot_id')
submission.shot_made_flag.value_counts()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_class_pred)

kobe2.period.value_counts()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
predictions = rf.predict(X_test)

metrics.accuracy_score(y_test, predictions)
confusion_matrix(y_test, predictions)

shot_predictions = rf.predict_proba(kobe_test[feature_cols])
kobe_test[feature_cols]
X[feature_cols]

made_predictions = pd.DataFrame(shot_predictions[:,1], columns = ['shot_made_flag'])
made_predictions.tail()
submission = pd.DataFrame(kobe_test.shot_id)
submission = submission.reset_index(drop=True)

submission.tail(50)

submission = submission.join(made_predictions)
submission = submission.set_index('shot_id')
submission.shot_made_flag.value_counts()
submission.tail()
submission.to_csv('kobe_submission.csv')
