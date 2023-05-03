import numpy as np
import pandas as pd
import csv
from sklearn.utils import Bunch

#Read the file
df = pd.read_csv('Datasets/Ebay Used Cars.csv')

# Drop unneeded columns
df = df.drop('Trim', axis=1)
df = df.drop('Engine', axis=1)
df = df.drop('yearsold', axis = 1)
df = df.drop('NumCylinders', axis = 1)

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
df = df.dropna()
df['zipcode'] = df['zipcode'].astype('int64')

# Added a new column for the age of the car
df['CarAge'] = 2023 - df['Year']
df.info()


#TODO
# Drop rows with mileage and price under a certain extreme amount
# Drop years which are not in a certain range (say 1920 - 2020). Drop yearsold column
# Drop rows where Make and Model have values which are extremely low frequency (under 20 occurrences)
# Instead of only keeping the first x most common frequencies of BodyType and DriveType, drop those entries
  # whose frequencies are outlier-range (say under 10 occurrences)
  # Possibly drop BodyType column entirely as this may be dependent variable on make and model
# How to handle electric cars? Perhaps drop the num-cylinders column entirely?
