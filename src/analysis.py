import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("data/TMDB_10000_Movies_Dataset.csv")

# Data Preparation
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['overview'] = df['overview'].fillna("")
df = df.drop_duplicates()

# Profiling Info
profiling_info = {
    "dtypes": df.dtypes.to_dict(),
    "missing_values": df.isnull().sum().to_dict(),
    "unique_counts": df.nunique().to_dict(),
    "stat_summary": df.describe().to_dict()
}
print(profiling_info)

# Clustering
features = df[['popularity', 'vote_average', 'vote_count']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Regression
X = df[['vote_average', 'vote_count']].dropna()
y = df.loc[X.index, 'popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df['pca1'] = pca_result[:,0]
df['pca2'] = pca_result[:,1]

# Visualizations
plt.figure(figsize=(8,6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='Set2')
plt.title("KMeans Clustering on TMDB Movies (PCA Reduced)")
plt.show()

sns.regplot(x="vote_average", y="popularity", data=df, scatter_kws={'alpha':0.5})
plt.title("Popularity vs Vote Average")
plt.show()

df['cluster'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6,6))
plt.title("Distribution of Movies by Cluster")
plt.ylabel("")
plt.show()
