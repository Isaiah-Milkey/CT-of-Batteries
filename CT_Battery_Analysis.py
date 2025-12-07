import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

#1) Load and Prepare Data
def load_and_merge_data(csv_paths):
    #Load multiple CSV files, pivot metrics, and merge into a single DataFrame from the battery data
    def pivot_csv(file_path):
        df = pd.read_csv(file_path)
        df = df[['sn', 'metric', 'mean']].copy()
        return df.pivot(index='sn', columns='metric', values='mean')

    pivoted_dfs = [pivot_csv(f) for f in csv_paths]
    merged_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'),
                       pivoted_dfs)
    merged_df.index.name = 'Serial_Number'
    merged_df['Cell_Number'] = merged_df.index.str.split('-').str[1].astype(int) #reduce serial number to just the number
    merged_df = merged_df.sort_values('Cell_Number').drop(columns=['Cell_Number'])
    return merged_df

csv_files = [
    r"F:\2170_anode\glimpse_inspection_data_2025-12-03-053508.csv",
    r"F:\2170_can\glimpse_inspection_data_2025-12-03-053611.csv",
    r"F:\2170_core\glimpse_inspection_data_2025-12-03-053711.csv",
    r"F:\2170_other\glimpse_inspection_data_2025-12-03-053822.csv"
]

battery_data = load_and_merge_data(csv_files)


#2) Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(battery_data)


#3) Principal Component Analysis (PCA)
pca = PCA(n_components=0.95)  # retain 95% of variance
principal_components = pca.fit_transform(scaled_features)

pca_columns = [f"PC_{i+1}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(principal_components, columns=pca_columns, index=battery_data.index)

#Explained variance plot (scree plot)
plt.figure(figsize=(8,4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_*100)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title("Scree Plot of PCA")
plt.tight_layout()
plt.show()

#PCA loadings for top features
loadings = pd.DataFrame(pca.components_.T, index=battery_data.columns, columns=pca_columns)
for i in range(min(2, len(pca_columns))):
    plt.figure(figsize=(10,4))
    loadings[f'PC_{i+1}'].abs().sort_values(ascending=False).head(10).plot(kind='bar')
    plt.ylabel("Absolute Loading")
    plt.title(f"Top Features Contributing to PC_{i+1}")
    plt.tight_layout()
    plt.show()


#4) Define Known Batches based on Box assignments
batch_categories = pd.Series(index=pca_df.index, dtype=str)

# Convert Index to Series for proper operations
cell_numbers = pd.Series(battery_data.index.str.split('-').str[1].astype(int), index=battery_data.index)

batch_categories[cell_numbers.between(1,130)] = 'Box 1 (Cells 1-130)'
batch_categories[cell_numbers.between(131,260)] = 'Box 2 (Cells 131-260)'
batch_categories[cell_numbers.between(261,390)] = 'Box 3 (Cells 261-390)'
batch_categories[cell_numbers.between(391,400)] = 'Small Plastic Holders (Cells 391-400)'
batch_categories[cell_numbers.between(401,450)] = 'Box 4 (Cells 401-450)'
batch_categories[cell_numbers.between(451,500)] = 'Box 5 (Cells 451-500)'

pca_df['Batch_Category'] = batch_categories

#5) K-Means Clustering and Best K Selection

# Use first 3 PCs for clustering
pcs_for_clustering = principal_components[:, :3]
candidate_ks = [2, 3, 4]

best_k = None
best_ari = -1
best_labels = None

for k in candidate_ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pcs_for_clustering)
    ari = adjusted_rand_score(pca_df['Batch_Category'], labels)
    if ari > best_ari:
        best_ari = ari
        best_k = k
        best_labels = labels

pca_df['Best_Cluster'] = best_labels
print(f"Selected Best K={best_k} based on ARI={best_ari:.4f}")


#6) Cluster-to-Batch Summary
cluster_summary_counts = pd.crosstab(pca_df['Batch_Category'], pca_df['Best_Cluster'])
cluster_summary_norm = pd.crosstab(pca_df['Batch_Category'], pca_df['Best_Cluster'], normalize='index')

print("\nCluster-to-Batch Counts:")
print(cluster_summary_counts)
print("\nNormalized Cluster-to-Batch Table:")
print(cluster_summary_norm)

#7) Cluster Feature Summary (Structural Insight)
feature_summary = battery_data.copy()
feature_summary['Cluster'] = pca_df['Best_Cluster']
cluster_feature_means = feature_summary.groupby('Cluster').mean()
print("\nMean Feature Values per Cluster:")
print(cluster_feature_means)


#8) Visualization: Clusters vs Batches (PC1 vs PC2)
plt.figure(figsize=(12,5))

#Best clusters
plt.subplot(1,2,1)
sns.scatterplot(x='PC_1', y='PC_2', hue='Best_Cluster', data=pca_df, palette='viridis', s=60)
plt.title(f"K-Means Clustering (K={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", loc='upper left')

#Known batches
plt.subplot(1,2,2)
sns.scatterplot(x='PC_1', y='PC_2', hue='Batch_Category', data=pca_df, palette='Set1', s=60)
plt.title("Known Manufacturing Batches")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Batch", loc='upper left')

plt.tight_layout()
plt.show()


#9) Top Distinguishing Features per Cluster
overall_mean = battery_data.mean()

#Compute absolute deviation from overall mean for each cluster
feature_deviation = cluster_feature_means.subtract(overall_mean).abs()

top_features_per_cluster = {}
for cluster in feature_deviation.index:
    top_features = feature_deviation.loc[cluster].sort_values(ascending=False).head(5)
    top_features_per_cluster[cluster] = top_features.index.tolist()
    print(f"\nCluster {cluster} - Top 5 distinguishing features:")
    print(top_features)

#Convert to DataFrame for cleaner presentation
top_features_df = pd.DataFrame.from_dict(top_features_per_cluster, orient='index', columns=[f"Feature_{i+1}" for i in range(5)])
print("\nSummary Table of Top Features per Cluster:")
print(top_features_df)
