import cProfile
import pandas as pd
from pstats import Stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from Functions import *
from LoadData import load_data

ONE_HOT = False

#Load data
df = load_data('Datasets/Ebay Used Cars.csv')

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
  
get_feature_importance('hreg', hreg, val_x, val_y, df)

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

get_feature_importance('rfreg', rfreg, val_x, val_y, df)

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

#Train and test ensembled HistGradientBoostingRegressor
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