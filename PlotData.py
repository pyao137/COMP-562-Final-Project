import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LoadData import load_data

df = load_data('Datasets/Ebay Used Cars.csv')

# Years vs Price Sold
var = 'Year'
data = pd.concat([df['pricesold'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 10))
fig = sns.boxplot(x=var, y="pricesold", data=data)
fig.axis(ymin=0, ymax=240000)
plt.xticks(rotation=90)
plt.show()

# Numbers of count of Make
var = "Make"
plt.figure(figsize=(20, 10))
sns.catplot(x=var, kind="count", palette="ch:.25", height=8, aspect=2, data=df);
plt.xticks(rotation=90)
plt.show()


# Mileage vs Price Sold
plt.figure(figsize=(20, 8))
hb = plt.hexbin(df['Mileage'], df['pricesold'], gridsize=(100, 50), cmap='viridis', mincnt=1, vmax=250)
plt.xlabel('Mileage')
plt.ylabel('Price Sold')
plt.title('Mileage vs Price Sold')
plt.xlim(0, 420000)
plt.ylim(0, 240000)
cb = plt.colorbar(hb, label='Number of Cars')
cb.set_ticks([0, 50, 100, 150, 200, 250])
plt.show()