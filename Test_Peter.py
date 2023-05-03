import numpy as np
import pandas as pd
import csv
from sklearn.utils import Bunch

df = pd.read_csv('Datasets/Ebay Used Cars.csv')

# Drop unneeded columns
df = df.drop('Trim', axis=1)
df = df.drop('Engine', axis=1)

# Clean up drive type and num-cylinders
df = df[df['DriveType'].str.len() == 3]
df['DriveType'] = df['DriveType'].str.lower()
top_drive_types = df['DriveType'].value_counts().head(7)
df = df[df['DriveType'].isin(top_drive_types.index)]
df = df[df['NumCylinders'] != 0]

# Find the top 20 variables in BodyType
df['BodyType'] = df['BodyType'].str.lower()
print(df['BodyType'].value_counts())
top_20 = df['BodyType'].value_counts().head(20)
sum_counts = top_20.sum()

# Clean up BodyType
df = df[df['BodyType'].isin(top_20.index)]

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
# Drop years which are not in a certain range (say 1950 - 2020). Drop yearsold not in reasonable range.
# Drop rows where Make and Model have values which are extremely low frequency (under 10 occurrences)
# Instead of only keeping the first x most common frequencies of BodyType and DriveType, drop those entries
  # whose frequencies are outlier-range (say under 10 occurrences)
  # Possibly drop BodyType column entirely as this may be dependent variable on make and model
  