import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

def one_hot(df, feature):
    dummies = pd.get_dummies(df[[feature]])
    res = pd.concat([df, dummies], axis=1)
    res = res.drop([feature], axis=1)
    return(res) 

def ordinal(df, feature):
  df[feature] = pd.factorize(df[feature])[0] + 1

def get_scores(model_name, model, val_x, val_y): #inputted model must be trained
  y_predictions = model.predict(val_x)
  mae = mean_absolute_error(val_y, y_predictions)
  print(f"{model_name} absolute mean error: {mae}")
  mse = mean_squared_error(val_y, y_predictions)
  rmse = math.sqrt(mse)
  print(f"{model_name} root mean squared error: {rmse}")  

def get_scores_ensembled(model_name, model_list, val_x, val_y): #inputted model must be trained. We assume there are 5 models in model list
  pred_list = []
  y_predictions = []
  for i in range(5): #There will be 5 models in the model_list
    pred_list.append(model_list[i].predict(val_x))
  pred_list = np.array(pred_list)
  y_predictions = np.mean(pred_list, axis = 0)
  mae = mean_absolute_error(val_y, y_predictions)
  print(f"{model_name} absolute mean error: {mae}")
  mse = mean_squared_error(val_y, y_predictions)
  rmse = math.sqrt(mse)
  print(f"{model_name} root mean squared error: {rmse}")  

def get_feature_importance(model_name, model, val_x, val_y, df):
  r = permutation_importance(model, val_x, val_y, n_repeats=10, random_state=0)
  print(f'Feature importances for {model_name}')
  for i in range(len(r.importances_mean)):
    print(f"  {df.columns[i + 1]}: {r.importances_mean[i]}")

def build_fit_ensemble_hreg(train_sets_x, train_sets_y):
  ensemble_hreg = []
  for i in range(len(train_sets_x)):
    hreg = HistGradientBoostingRegressor(
      max_iter = 1000,
      loss = 'squared_error'
    )
    hreg.fit(train_sets_x[i], train_sets_y[i])
    ensemble_hreg.append(hreg)
  return ensemble_hreg



