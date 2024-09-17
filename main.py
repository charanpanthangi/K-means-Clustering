# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load dataset
data = pd.read_csv('your_data.csv')

# Univariate Analysis
def univariate_analysis(data):
    print("Univariate Analysis:")
    for column in data.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.show()

# Bivariate Analysis (Pair plot)
def bivariate_analysis(data):
    print("Bivariate Analysis:")
    sns.pairplot(data)
    plt.show()

# Preprocess data
features = data.drop(columns=['target'])  # Replace 'target' with any non-feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# K-Means Clustering
def fit_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

# Fit the K-Means model
n_clusters = 3  # You can choose the number of clusters based on your requirement
kmeans_model = fit_kmeans(X_scaled, n_clusters)

# Predict cluster labels
cluster_labels = kmeans_model.predict(X_scaled)

# Add cluster labels to the original data
data['Cluster'] = cluster_labels

# Save the trained K-Means model
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans_model, file)

# Load the saved K-Means model and predict cluster labels for new data
with open('kmeans_model.pkl', 'rb') as file:
    loaded_kmeans = pickle.load(file)

# Example of passing new input data to the model
new_input = np.array([[1.5, 2.5, 3.0]])  # Replace with your new input data
new_input_scaled = scaler.transform(new_input)
cluster_prediction = loaded_kmeans.predict(new_input_scaled)
print("Cluster Prediction for new input:", cluster_prediction)

# Call the analysis functions
univariate_analysis(data)
bivariate_analysis(data)
