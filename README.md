# titanic_ml
Basic ML with Titanic dataset


model.py:

Basic set up DecisionTree, no feature engineering done.
Replaced NA's with 1 (not ideal)

Model Score: 0.78027



better_model.py:

Still basic set up DecisionTree, some feature engineering done. Add 'FamTot' as an int for number of family members on board.
I also one hot encoded Family's last names because I assumed members within the family may be more correlated in terms of
outcome.

Model Score: 0.85650
