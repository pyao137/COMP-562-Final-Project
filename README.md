# COMP-562-Final-Project

Dataset: https://www.kaggle.com/datasets/tsaustin/us-used-car-sales-data 

DROPPING NUM-CYLINDERS
Setup: validation data contains entries from 58000 to end
Current scores (on final validation data run)
HistGradientBoostingRegressor (One-model fit on the first 58000): 0.6453008002033734
GradientBoostingRegressor (One-model fit on the first 58000): 0.5538657719295113
Ridge Regression (One-model fit on the first 58000): 0.4634632608660838
LinearRegression (One-model fit on the first 58000): 0.43022578978416937

NOT DROPPING NUM-CYLINDERS
Setup: validation data contains entries from 52000 to end
Current scores (on final validation data run)
HistGradientBoostingRegressor (One-model fit on the first 58000): 0.6722605456141024
GradientBoostingRegressor (One-model fit on the first 58000): 0.5822480201354919
LinearRegression (One-model fit on the first 58000): -2736835676.72514 (Low score likely due to the fact that num-cylinders is highly correlated with make and model)
    See this for more: https://medium.com/analytics-vidhya/the-pitfalls-of-linear-regression-and-how-to-avoid-them-b93626e1a020 
Ridge Regression (One-model fit on the first 58000): -3077859404.0927434

Trial 2 not dropping num cylinders error: /usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_ridge.py:216: LinAlgWarning: Ill-conditioned matrix (rcond=6.80874e-20): result may not be accurate.
  return linalg.solve(A, Xy, assume_a="pos", overwrite_a=True).T 

  This issue is for the ridge regression model

TODO:
1. Figure out how to get root mean square error and mean absolute error in scoring
2. Ensemble Either randforest or histgradient and train over epochs
3. Find alternatives to one-hot encoding of categorical variables (as too many columns)
4. Possibly use grid search to tune hyperparmeters for the decision tree models
5. Neural Net?
