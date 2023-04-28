import numpy as np
import pandas as pd
import csv
from sklearn.utils import Bunch

#Clean up zipcode
df = pd.read_csv('Datasets/Ebay Used Cars.csv')
ndf = df[df['zipcode'].str.len() == 5]
ndf = ndf[ndf['zipcode'].str[2:5] != '***']
ndf['zipcode'] = ndf['zipcode'].str[:-2]
ndf['zipcode'] = pd.to_numeric(ndf['zipcode'], downcast = 'integer')
print(ndf.head(1000))