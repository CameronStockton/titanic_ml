# titanic_ml
Basic ML with Titanic dataset\n
\n
model.py: \n
Basic set up DecisionTree, no feature engineering done. \n
Replaced NA's with 1 (not ideal)\n
\n
Model Score: 0.78027\n
\n
\n
better_model.py\n
Still basic set up DecisionTree, some feature engineering done. Add 'FamTot' as an int for number of family members on board.\n
I also one hot encoded Family's last names because I assumed members within the family may be more correlated in terms of
outcome.\n
\n
Model Score: 0.85650
