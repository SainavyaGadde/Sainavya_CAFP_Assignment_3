import pandas as pd
import errors

# Read the dataset
df1 = pd.read_csv('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5447781.csv', skiprows=4)

# Display the head of the dataset
df1.head()



# Display the shape of the dataset
df1.shape


import pandas as pd

# Read the dataset
df2 = pd.read_csv('API_NY.GDP.PETR.RT.ZS_DS2_en_csv_v2_5363502.csv', skiprows=4)

# Display the head of the dataset
df2.head()



# Display the shape of the dataset
df2.shape


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5447781.csv", skiprows=4)

# Select relevant columns and drop missing values
df = df[["Country Name", "2019"]].dropna()

# Rename columns
df.columns = ["Country", "GDP_per_capita"]

# Set index to country name
df.set_index("Country", inplace=True)

# Remove rows with invalid GDP values (negative or zero)
df = df[df["GDP_per_capita"] > 0]

# Log-transform the GDP values to reduce skewness
df["GDP_per_capita"] = np.log(df["GDP_per_capita"])

# Standardize the data using z-score normalization
df = (df - df.mean()) / df.std()

# Save the cleaned dataset to a new file
df.to_csv("CLUSTER_dataset.csv")


import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("CLUSTER_dataset.csv")

# Select 4 countries
countries = ['Bahrain', 'Kuwait', 'Oman', 'Qatar']

# Filter the dataset for the selected countries
df_countries = df.loc[df["Country"].isin(countries)]

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(df_countries["Country"], df_countries["GDP_per_capita"])
plt.xlabel("Country")
plt.ylabel("GDP per capita (Standardized)")
plt.title("GDP per Capita for sample Gulf Countries")
plt.xticks(rotation=45)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the cleaned dataset
df = pd.read_csv("CLUSTER_dataset.csv")

# Extract the GDP per capita column and normalize the data
X = df['GDP_per_capita'].values.reshape(-1, 1)
X_norm = (X - X.mean()) / X.std()

# Define the range of number of clusters to try
n_clusters_range = range(2, 11)

# Iterate over the number of clusters and compute the silhouette score
silhouette_scores = []
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_norm)
    silhouette_scores.append(silhouette_score(X_norm, labels))

# Plot the silhouette scores
plt.plot(n_clusters_range, silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis: Optimal Number of Clusters")
plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the cleaned dataset
df = pd.read_csv('CLUSTER_dataset.csv')

# Extract GDP per capita column and normalize
X = df['GDP_per_capita'].values.reshape(-1, 1)
X_norm = StandardScaler().fit_transform(X)

# Perform Gaussian Mixture Model clustering with n_clusters=4
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_norm)
df['Cluster'] = gmm.predict(X_norm)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red', 'green', 'blue', 'orange']
for i in range(4):
    cluster_data = df[df['Cluster'] == i]
    scatter = ax.scatter(cluster_data.index, cluster_data['GDP_per_capita'],
                         color=colors[i], label=f'Cluster {i+1}')
plt.xticks(np.arange(0, df.shape[0], 50), np.arange(0, df.shape[0], 50), fontsize=12)
plt.xlabel('Country Index', fontsize=14)
plt.ylabel('GDP per capita', fontsize=14)
plt.title('Gaussian Mixture Model Clustering Results', fontsize=16)
ax.legend(fontsize=12)

# Add annotation for the cluster centers
centers = gmm.means_
for i, center in enumerate(centers):
    ax.annotate(f'Cluster {i+1} center: {center[0]:,.2f}', xy=(1, center[0]), xytext=(6, 0),
                textcoords="offset points", ha='left', va='center', fontsize=12, color=colors[i])

plt.show()


# print countries in each cluster in a table
for i in range(4):
    print(f'Cluster {i+1}:')
    cluster_data = df[df['Cluster']==i]
    cluster_table = pd.DataFrame({'Country': cluster_data['Country'].values})
    display(cluster_table)


import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('API_NY.GDP.PETR.RT.ZS_DS2_en_csv_v2_5363502.csv', skiprows=4)

# Select only the necessary data for fitting analysis
df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', *df.columns[-32:-1]]]  # Update the column range

# Rename columns to simpler names
df.columns = ['Country', 'Code', 'Indicator', 'IndicatorCode', *range(1990, 2021)]  

# Melt the DataFrame to transform the columns into rows
df_melted = pd.melt(df, id_vars=['Country', 'Code', 'Indicator', 'IndicatorCode'], var_name='Year', value_name='Value')

# Drop rows with missing values
df_cleaned = df_melted.dropna()

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('FITTING_data.csv', index=False)


import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df_cleaned = pd.read_csv('FITTING_data.csv')

# Select the Gulf countries (e.g., Bahrain, Kuwait, Oman, Qatar)
gulf_countries = ['Bahrain', 'Kuwait', 'Oman', 'Qatar']
df_gulf = df_cleaned[df_cleaned['Country'].isin(gulf_countries)]

# Create a line chart
fig, ax = plt.subplots(figsize=(12, 8))

for country in gulf_countries:
    country_data = df_gulf[df_gulf['Country'] == country]
    ax.plot(country_data['Year'], country_data['Value'], label=country)

plt.xlabel('Year', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('Gulf Countries - GDP Petroleum Rents', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()


import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('FITTING_data.csv')

# Filter data for Qatar
qatar_data = df[df['Country'] == 'Qatar']

# Extract the necessary columns
years = qatar_data['Year'].values
values = qatar_data['Value'].values

# Define the exponential function
def exponential_model(x, a, b, c):
    return a * np.exp(b * (x - years[0])) + c

# Perform curve fitting
popt, pcov = curve_fit(exponential_model, years, values)

# Generate predictions for future years
future_years = np.arange(years.min(), years.max() + 21)  # Predict for 20 additional years
predicted_values = exponential_model(future_years, *popt)

# Calculate confidence ranges
sigma = np.sqrt(np.diag(pcov))
lower, upper = errors.err_ranges(future_years, exponential_model, popt, sigma)

# Plot the best fitting function and confidence range
plt.figure(figsize=(12, 8))
plt.plot(years, values, 'ko', label='Actual Data')
plt.plot(future_years, predicted_values, 'r-', label='Best Fitting Function')
plt.fill_between(future_years, lower, upper, color='gray', alpha=0.4, label='Confidence Range')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('Exponential Model Fit for Qatar', fontsize=16)
plt.legend(fontsize=12)
plt.ylim(-100, 100)  # Set the y-axis limits to -100 and 100
plt.grid(True)
plt.show()


# Print the predicted values for the future years
print('Predicted Values:')
for year, value in zip(future_years, predicted_values):
    print(f'{year}: {value:.2f}')
