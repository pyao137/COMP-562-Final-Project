import numpy as np
import pandas as pd
import csv
from sklearn.utils import Bunch
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import skew

plt.style.use('ggplot')
sns.set()


def one_hot(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return (res)


# Read the file
df = pd.read_csv('Datasets/Ebay Used Cars.csv')

# Drop unneeded columns
df = df.drop('Trim', axis=1)
df = df.drop('Engine', axis=1)
df = df.drop('yearsold', axis=1)
df = df.drop('NumCylinders', axis=1)
df = df.drop('ID', axis=1)

# Clean up drive type and num-cylinders (drop 0 cylinders cars and drop outlier DriveTypes)
df = df[df['DriveType'].str.len() == 3]
df['DriveType'] = df['DriveType'].str.lower()
drive_types = df['DriveType'].value_counts().gt(10)
df = df.loc[df['DriveType'].isin(drive_types[drive_types].index)]
# df = df[df['NumCylinders'] != 0]

# Clean up BodyType (Drop outliers with freq. under 50)
df['BodyType'] = df['BodyType'].str.lower()
body_types = df['BodyType'].value_counts().gt(50)
df = df.loc[df['BodyType'].isin(body_types[body_types].index)]

# Clean up model (Drop outliers with freq. under 20)
df['Model'] = df['Model'].str.lower()
models = df['Model'].value_counts().gt(20)
df = df.loc[df['Model'].isin(models[models].index)]

# Clean up Make (Drop outliers with freq. under 20)
df['Make'] = df['Make'].str.lower()
makes = df['Make'].value_counts().gt(20)
df = df.loc[df['Make'].isin(makes[makes].index)]

# Clean up numerical values
df = df[df['Mileage'] > 100]
df = df[df['Mileage'] < 400000]
df = df[df['Year'] > 1920]
df = df[df['pricesold'] > 200]

# Clean up zipcode and drop nulls
ndf = df[df['zipcode'].str.len() == 5]
ndf = ndf[ndf['zipcode'].str[2:5] != '***']
ndf['zipcode'] = ndf['zipcode'].str[:-2]
df['zipcode'] = pd.to_numeric(ndf['zipcode'], downcast='integer').astype('Int64')
df = df.dropna()  # Drop nulls
df['zipcode'] = df['zipcode'].astype('int64')

# Added a new column for the age of the car
df['CarAge'] = 2023 - df['Year']

# #One-hot encode all categorical variables
# one_hot_features = ['Make', 'Model', 'BodyType', 'DriveType']
# for feature in one_hot_features:
#     df = one_hot(df, feature)
#
#
# #Randomize rows and split dataset into train, test, and validation data
# df = df.sample(frac=1)
# df.info()
#
# train_df = df.iloc[:55000,:]
# #test_df = df.iloc[50000:59000,:]
# val_df = df.iloc[55000:,:]
#
# #Train the model
# df_x = df.drop("pricesold", axis = 1).to_numpy()
# df_y = (df["pricesold"]).to_numpy()
# train_x = train_df.drop("pricesold", axis = 1).to_numpy()
# train_y = (train_df["pricesold"]).to_numpy()
# val_x = val_df.drop("pricesold", axis = 1).to_numpy()
# val_y = (val_df["pricesold"]).to_numpy()
# regressor = HistGradientBoostingRegressor()
# regressor.fit(df_x, df_y)
# print(regressor.score(df_x, df_y))

# f, ax = plt.subplots(figsize=(15, 8))
# sns.displot(df['pricesold'])
# plt.xlim([0, 160000])

plt.style.use('ggplot')
# select all quantitative columns for checking the spread
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(20, 25))

for i, variable in enumerate(numeric_columns):
    plt.subplot(10, 3, i + 1)

    sns.distplot(df[variable], kde=False, color='blue')
    plt.tight_layout()
    plt.title(variable)
plt.show()

cat_columns = ['Make', 'Mileage', 'DriveType', 'pricesold']  # cars.select_dtypes(
# exclude=np.number).columns.tolist()

plt.figure(figsize=(15, 21))

for i, variable in enumerate(cat_columns):
    plt.subplot(4, 2, i + 1)
    order = df[variable].value_counts(ascending=False).index
    ax = sns.countplot(x=df[variable], data=df, order=order, palette='viridis')
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / len(df[variable]))
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        plt.annotate(percentage, (x, y), ha='center')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title(variable)

plt.show()
# TODO
# Drop rows with mileage and price under a certain extreme amount
# Drop years which are not in a certain range (say 1920 - 2020). Drop yearsold column
# Drop rows where Make and Model have values which are extremely low frequency (under 20 occurrences)
# Instead of only keeping the first x most common frequencies of BodyType and DriveType, drop those entries
# whose frequencies are outlier-range (say under 10 occurrences)
# Possibly drop BodyType column entirely as this may be dependent variable on make and model
# How to handle electric cars? Perhaps drop the num-cylinders column entirely?
