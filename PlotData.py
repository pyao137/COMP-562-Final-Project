import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from LoadData import load_data
import numpy as np

df = load_data('Datasets/Ebay Used Cars.csv')

# Years vs Price Sold
var = 'Year'
data = pd.concat([df['pricesold'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 8))
fig = sns.boxplot(x=var, y="pricesold", data=data, ax=ax)
fig.axis(ymin=0, ymax=240000)

unique_years = sorted(data[var].unique())
interval = 4  # Change this value to adjust the interval
xtick_positions = np.arange(0, len(unique_years), interval)
xtick_labels = [unique_years[i] for i in xtick_positions]

ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, rotation=90)

plt.show()

# Numbers of count of Make
var = "Make"
plt.figure(figsize=(20, 8))

# Aggregate and sort data by 'Make' frequency
make_counts = df[var].value_counts().sort_values(ascending=False)
sorted_make = pd.DataFrame(make_counts).reset_index()
sorted_make.columns = [var, 'count']

# Create the sorted catplot
sns.catplot(x=var, y='count', kind="bar", palette="ch:.25", height=8, aspect=2, data=sorted_make)

plt.xticks(rotation=90)
plt.show()

# Top 10 numbers of count of Make
var = "Make"
plt.figure(figsize=(20, 8))

# Aggregate and sort data by 'Make' frequency
make_counts = df[var].value_counts().sort_values(ascending=False)
sorted_make = pd.DataFrame(make_counts).reset_index()
sorted_make.columns = [var, 'count']

# For only the top 10 frequencies of Make
top_10_make = sorted_make.head(10)

# Create the sorted catplot for top 10 Make
sns.catplot(x=var, y='count', kind="bar", palette="ch:.25", height=8, aspect=2, data=top_10_make)

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

# Frequency of the Year
year_ranges = [(1920, 1930), (1930, 1940), (1940, 1950), (1950, 1960), (1960, 1970), (1970, 1980), (1980, 1990),
               (1990, 2000), (2001, 2010), (2011, 2020)]
labels = ['1920-1930', '1930-1940', '1940-1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990', '1990-2000',
          '2001-2010', '2011-2020']
counts = []

for start_year, end_year in year_ranges:
    count = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].shape[0]
    counts.append(count)

plt.figure(figsize=(14, 6))
plt.bar(labels, counts)
plt.xlabel('Year Ranges')
plt.ylabel('Number of Cars Sold')
plt.title('Frequency of Cars Sold by Year Range')
plt.xticks(rotation=45)
plt.show()
