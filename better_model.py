import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['Sex'] = train_df['Sex'].replace({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].replace({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
test_df['Embarked'] = test_df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
train_df['Age'] = train_df['Age'].fillna(-1)
test_df['Age'] = test_df['Age'].fillna(-1)
train_df['Embarked'] = train_df['Embarked'].fillna(-1)
test_df['Embarked'] = test_df['Embarked'].fillna(-1)

#Now let's do some feature engineering
#Create # in family column
train_df['FamTot'] = train_df['SibSp'] + train_df['Parch']
test_df['FamTot'] = test_df['SibSp'] + test_df['Parch']
#Now i want to make a family name column and one hot encode that so
#we can group by family. My hunch is that family members may live/die
#together more often than not
train_df['FamName'] = train_df['Name'].str.replace(',', '').str.split(' ').str[0]
test_df['FamName'] = test_df['Name'].str.replace(',', '').str.split(' ').str[0]

train_one_hot_encoded = pd.get_dummies(train_df['FamName'], prefix='FamName')
train_df = pd.concat([train_df, train_one_hot_encoded], axis=1)
test_one_hot_encoded = pd.get_dummies(test_df['FamName'], prefix='FamName')
test_df = pd.concat([test_df, test_one_hot_encoded], axis=1)


train_df_x, train_df_y = train_df.drop(['PassengerId', 'Survived', 'Name', 'Cabin', 'Ticket', 'FamName'], axis=1), train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(train_df_x, train_df_y, random_state=0)

clf = tree.DecisionTreeClassifier(random_state=2).fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
print('The accuracy score of this model is: ' + str(acc))
print('The precision score of this model is: ' + str(precision.mean()))
print('The recall score of this model is: ' + str(recall.mean()))

disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        cmap=plt.cm.Blues
    )

print(disp.confusion_matrix)

plt.show()