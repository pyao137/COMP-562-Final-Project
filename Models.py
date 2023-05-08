import math
import cProfile
import pandas as pd
import numpy as np
from pstats import Stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error ,mean_absolute_error
from sklearn.inspection import permutation_importance

DROP_CYL = False #DON'T CHANGE
ONE_HOT = False

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

def get_feature_importance(model_name, model, val_x, val_y):
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

#Read the file
df = pd.read_csv('gdrive/My Drive/Ebay Used Cars.csv')
etest = df[df['Engine'].str.lower() == 'electric'] #2 no non-tesla makes have more than 15 sales as marked electric cars:

# Drop unneeded columns
df = df.drop('Trim', axis=1)
df = df.drop('Engine', axis=1)
df = df.drop('yearsold', axis = 1)
df = df.drop('ID', axis = 1)

#Possible: Drop num cylinders
if (DROP_CYL):
   df = df.drop('NumCylinders', axis = 1)

# Clean up drive type (drop outlier DriveTypes)
df = df[df['DriveType'].str.len() == 3]
df['DriveType'] = df['DriveType'].str.lower()
drive_types = df['DriveType'].value_counts().gt(10)
df = df.loc[df['DriveType'].isin(drive_types[drive_types].index)]

#Clean up BodyType (Drop outliers with freq. under 50)
df['BodyType'] = df['BodyType'].str.lower()
body_types = df['BodyType'].value_counts().gt(50)
df = df.loc[df['BodyType'].isin(body_types[body_types].index)]

#Clean up model (Drop outliers with freq. under 20)
df['Model'] = df['Model'].str.lower()
models = df['Model'].value_counts().gt(20)
df = df.loc[df['Model'].isin(models[models].index)]

#Clean up Make (Drop outliers with freq. under 20)
df['Make'] = df['Make'].str.lower()
makes = df['Make'].value_counts().gt(20)
df = df.loc[df['Make'].isin(makes[makes].index)]

#Clean up numerical values
df = df[df['Mileage'] > 100]
df = df[df['Mileage'] < 400000]
df = df[df['Year'] > 1920]
df = df[df['Year'] < 2021]
df = df[df['pricesold'] > 200]

#Clean up num-cylinders if not dropped
if (DROP_CYL == False):
  df = df[ (df['NumCylinders'] != 0) | (df['Make'] == 'tesla')]

# Clean up zipcode and drop nulls
ndf = df[df['zipcode'].str.len() == 5]
ndf = ndf[ndf['zipcode'].str[2:5] != '***']
ndf['zipcode'] = ndf['zipcode'].str[:-2]
df['zipcode'] = pd.to_numeric(ndf['zipcode'], downcast='integer').astype('Int64')
df = df.dropna() #Drop nulls
df['zipcode'] = df['zipcode'].astype('int64')

#Encode categorical variables
cat_feat = ['Make', 'Model', 'BodyType', 'DriveType']
if (ONE_HOT):
  for feature in cat_feat: 
    df = one_hot(df, feature)
else:
  for feature in cat_feat: 
    ordinal(df, feature)

#Randomize rows and split dataset into train and validation data
df = df.sample(frac=1)

if ('NumCylinders' in df):
  train_df = df.iloc[:53000,:]
  val_df = df.iloc[53000:,:]
else:
  train_df = df.iloc[:58000,:]
  val_df = df.iloc[58000:,:]

train_x = train_df.drop("pricesold", axis = 1).to_numpy()
train_y = (train_df["pricesold"]).to_numpy()
val_x = val_df.drop("pricesold", axis = 1).to_numpy()
val_y = (val_df["pricesold"]).to_numpy()

#Split train data into 5 folds (for use on the ensembling technique):
train_sets_x = []
train_sets_y = []
for i in range(5):
  new_df = train_df.drop(train_df.index[i*10500:(i+1)*10500], inplace = False)
  train_sets_y.append((new_df["pricesold"]).to_numpy())
  train_sets_x.append(new_df.drop("pricesold", axis = 1).to_numpy())
print(len(train_sets_x[0]))
print(df.info())

#Train, test and get feature importance for HistGradientBoostingRegressor
with cProfile.Profile() as pr:
  hreg = HistGradientBoostingRegressor(max_iter = 1000, loss = 'squared_error')
  hreg.fit(train_x ,train_y)
  print("time to fit:")
  stats = Stats(pr)
  stats.sort_stats('tottime').print_stats(5)

with cProfile.Profile() as pr:
    get_scores("hreg", hreg, val_x, val_y)
    print("time to run:")
    stats = Stats(pr)
    stats.sort_stats('tottime').print_stats(5)
  
get_feature_importance('hreg', hreg, val_x, val_y)

#Train, test, and get feature importance for RandomForest
with cProfile.Profile() as pr:
  rfreg = RandomForestRegressor(n_estimators = 100, min_samples_split = 3, min_samples_leaf = 3, random_state = 101)
  rfreg.fit(train_x, train_y)
  print("time to fit:")
  stats = Stats(pr)
  stats.sort_stats('tottime').print_stats(5)

with cProfile.Profile() as pr:
  get_scores("rfreg", rfreg, val_x, val_y)
  print("time to run:")
  stats = Stats(pr)
  stats.sort_stats('tottime').print_stats(5)

get_feature_importance('rfreg', rfreg, val_x, val_y)

#Train, test, and get coefficients for LinearRegression
with cProfile.Profile() as pr:
  linreg = LinearRegression()
  linreg.fit(train_x, train_y)
  print("time to fit:")
  stats = Stats(pr)
  stats.sort_stats('tottime').print_stats(5)

with cProfile.Profile() as pr:
  get_scores("linreg", linreg, val_x, val_y)
  print("time to run:")
  stats = Stats(pr)
  stats.sort_stats('tottime').print_stats(5)

print("Coefficients for linreg")
for i in range(len(linreg.coef_)):
  print(f'{df.columns[i + 1]}: {linreg.coef_[i]}')

#Train and test ensembled hreg
with cProfile.Profile() as pr:
  ensemble_hreg = build_fit_ensemble_hreg(train_sets_x, train_sets_y)
  print("time to fit:")
  stats = Stats(pr)
  stats.sort_stats('tottime').print_stats(5)

with cProfile.Profile() as pr:
  get_scores_ensembled("ens-hreg", ensemble_hreg, val_x, val_y)
  print("time to run:")
  stats = Stats(pr)
  stats.sort_stats('tottime').print_stats(5)