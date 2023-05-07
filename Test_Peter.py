import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def one_hot(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

#Read the file
df = pd.read_csv('Datasets/Ebay Used Cars.csv')

# Drop unneeded columns
df = df.drop('Trim', axis=1)
df = df.drop('Engine', axis=1)
df = df.drop('yearsold', axis = 1)
df = df.drop('NumCylinders', axis = 1)
df = df.drop('ID', axis = 1)

# Clean up drive type and num-cylinders (drop 0 cylinder cars and drop outlier DriveTypes)
df = df[df['DriveType'].str.len() == 3]
df['DriveType'] = df['DriveType'].str.lower()
drive_types = df['DriveType'].value_counts().gt(10)
df = df.loc[df['DriveType'].isin(drive_types[drive_types].index)]
#df = df[df['NumCylinders'] != 0]

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
df = df[df['pricesold'] > 200]

# Clean up zipcode and drop nulls
ndf = df[df['zipcode'].str.len() == 5]
ndf = ndf[ndf['zipcode'].str[2:5] != '***']
ndf['zipcode'] = ndf['zipcode'].str[:-2]
df['zipcode'] = pd.to_numeric(ndf['zipcode'], downcast='integer').astype('Int64')
df = df.dropna() #Drop nulls
df['zipcode'] = df['zipcode'].astype('int64')

# Added a new column for the age of the car
df['CarAge'] = 2023 - df['Year']

#One-hot encode all categorical variables
one_hot_features = ['Make', 'Model', 'BodyType', 'DriveType']
for feature in one_hot_features:
    df = one_hot(df, feature)


#Randomize rows and split dataset into train, test, and validation data
df = df.sample(frac=1)
df.info()
train_df = df.iloc[:55000,:]
val_df = df.iloc[55000:,:]

#Train the model
df_x = df.drop("pricesold", axis = 1).to_numpy()
df_y = (df["pricesold"]).to_numpy()
train_x = train_df.drop("pricesold", axis = 1).to_numpy()
train_y = (train_df["pricesold"]).to_numpy()
val_x = val_df.drop("pricesold", axis = 1).to_numpy()
val_y = (val_df["pricesold"]).to_numpy()
regressor = HistGradientBoostingRegressor()
regressor.fit(train_x, train_y)
print(regressor.score(val_x, val_y))

#linreg = LinearRegression().fit(df_x, df_y)
#print(linreg.score(df_x, df_y))



#TODO
# Drop rows with mileage and price under a certain extreme amount
# Drop years which are not in a certain range (say 1920 - 2020). Drop yearsold column
# Drop rows where Make and Model have values which are extremely low frequency (under 20 occurrences)
# Instead of only keeping the first x most common frequencies of BodyType and DriveType, drop those entries
  # whose frequencies are outlier-range (say under 10 occurrences)
  # Possibly drop BodyType column entirely as this may be dependent variable on make and model
# How to handle electric cars? Perhaps drop the num-cylinders column entirely?
