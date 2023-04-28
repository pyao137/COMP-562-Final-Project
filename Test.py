import numpy as np
import pandas as pd
import csv
from sklearn.utils import Bunch

#Clean up zipcode
df = pd.read_csv('Datasets/Ebay Used Cars.csv')
ndf = df[df['zipcode'].str.len() == 5]
ndf = ndf[ndf['zipcode'].str[2:5] != '***']
ndf['zipcode'] = ndf['zipcode'].str[:-2]
df['zipcode'] = pd.to_numeric(ndf['zipcode'], downcast = 'integer')

#Drop unneeded columns
df = df.drop('Trim', axis=1)
df = df.drop('Engine', axis=1)

#Clean up drive type and num-cylinders
df = df[df['DriveType'].str.len() == 3]
df = df[df['NumCylinders'] != 0]
print(len(df))