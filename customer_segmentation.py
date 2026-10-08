import matplotlib

matplotlib.use("Agg")  # save plots to file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ---------------------------------------------------------------------------
# Step 1 / Question 2a: data preparation and choice of k
# ---------------------------------------------------------------------------
# Prepare the features (drop the ID, standardize), then compare WCSS and
# silhouette across several values of k to choose how many clusters to use

print("=" * 70)

df = pd.read_csv("data/customers.csv")
feature_names = ["Age", "Annual Spending ($)", "Purchases per Month"]
X = df[feature_names]  # Customer ID only labels a row, so it is not a feature

# Spending spans 200-900 while purchases span 1-7: unscaled, spending would
# dominate every distance calculation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaling_rows = []
for column_index, feature_name in enumerate(feature_names):
    scaling_rows.append(
        [
            feature_name,
            f"{X[feature_name].min()} - {X[feature_name].max()}",
            round(X_scaled[:, column_index].mean(), 2) + 0.0,  # + 0.0 turns -0.0 into 0.0
            round(X_scaled[:, column_index].std(), 2),
        ]
    )
print(
    tabulate(
        scaling_rows,
        headers=["Feature", "Raw range", "Scaled mean", "Scaled std"],
        tablefmt="fancy_grid",
    )
)

# WCSS and silhouette for each candidate k
candidate_ks = range(1, 7)
wcss_values = []
k_rows = []
for k_value in candidate_ks:
    model = KMeans(n_clusters=k_value, random_state=42, n_init=10)  # fixed seed for reproducible results
    model.fit(X_scaled)
    wcss_values.append(model.inertia_)
    # silhouette needs at least 2 clusters
    silhouette = silhouette_score(X_scaled, model.labels_) if k_value > 1 else None
    k_rows.append(
        [k_value, round(model.inertia_, 3), "n/a" if silhouette is None else round(silhouette, 3)]
    )
print(
    tabulate(
        k_rows,
        headers=["k", "WCSS", "Silhouette"],
        tablefmt="fancy_grid",
    )
)

plt.figure(figsize=(6, 4))
plt.plot(list(candidate_ks), wcss_values, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow method")
plt.tight_layout()
plt.savefig("elbow_method.png")
# plt.show()

K = 3  # elbow at k=3; chosen segments are also easy to interpret
print(f"Chosen number of clusters: k = {K}")

# ---------------------------------------------------------------------------
# Step 2 / Question 2b: K-means clustering
# ---------------------------------------------------------------------------
# Group the customers with K-means using the chosen k, then summarize each
# cluster in the original units (the helper is reused by later sections)

def summarize_clusters(customers, labels):
    """Print the size and average raw features of each cluster (noise = -1)."""
    summary_rows = []
    for cluster_label in sorted(set(labels)):
        members = customers[labels == cluster_label]
        summary_rows.append(
            [
                "Noise" if cluster_label == -1 else f"Cluster {cluster_label}",
                len(members),
                ", ".join(members["Customer ID"]),
                round(members["Age"].mean(), 1),
                round(members["Annual Spending ($)"].mean(), 1),
                round(members["Purchases per Month"].mean(), 1),
            ]
        )
    print(
        tabulate(
            summary_rows,
            headers=["Group", "Size", "Customers", "Avg age", "Avg spending ($)", "Avg purchases/month"],
            tablefmt="fancy_grid",
        )
    )


print("=" * 70)
print("QUESTION 2b: K-MEANS CLUSTERING")
print("=" * 70)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)  # fixed seed for reproducible results
kmeans_labels = kmeans.fit_predict(X_scaled)

print(f"Iterations until the centroids stopped moving: {kmeans.n_iter_}")
print(f"WCSS: {kmeans.inertia_:.3f}")
summarize_clusters(df, kmeans_labels)

plt.figure(figsize=(6, 4))
plt.scatter(df["Annual Spending ($)"], df["Purchases per Month"], c=kmeans_labels)
for row_index, customer_id in enumerate(df["Customer ID"]):
    plt.annotate(customer_id, (df["Annual Spending ($)"][row_index], df["Purchases per Month"][row_index]))
plt.xlabel("Annual Spending ($)")
plt.ylabel("Purchases per Month")
plt.title(f"K-means clusters (k = {K})")
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
# plt.show()

# ---------------------------------------------------------------------------
# Step 3 / Question 2c: hierarchical clustering and DBSCAN
# ---------------------------------------------------------------------------
# Hierarchical: merge the closest groups step by step, then cut the tree into k
# clusters. DBSCAN: grow clusters from dense areas; isolated points become noise
print("=" * 70)
print("QUESTION 2c: HIERARCHICAL CLUSTERING AND DBSCAN")
print("=" * 70)

print("Hierarchical clustering (Ward linkage, cut into k clusters)")
merge_tree = linkage(X_scaled, method="ward")
# renumber so cluster labels start at 0 in order of first appearance
hierarchical_labels = pd.factorize(fcluster(merge_tree, t=K, criterion="maxclust"))[0]
summarize_clusters(df, hierarchical_labels)

plt.figure(figsize=(7, 4))
dendrogram(merge_tree, labels=df["Customer ID"].tolist())
plt.ylabel("Merge distance (Ward)")
plt.title("Hierarchical clustering dendrogram")
plt.tight_layout()
plt.savefig("dendrogram.png")
# plt.show()

print("DBSCAN (eps = 0.8, min_samples = 3)")
dbscan_labels = DBSCAN(eps=0.8, min_samples=3).fit_predict(X_scaled)
summarize_clusters(df, dbscan_labels)

print("Side-by-side comparison of the three methods")
comparison = pd.DataFrame(
    {
        "Customer": df["Customer ID"],
        "K-means": kmeans_labels,
        "Hierarchical": hierarchical_labels,
        "DBSCAN": pd.Series(dbscan_labels).replace(-1, "noise"),
    }
)
print(tabulate(comparison, headers="keys", tablefmt="fancy_grid", showindex=False))

# ---------------------------------------------------------------------------
# Step 4 / Question 3b: silhouette scores for all three methods
# ---------------------------------------------------------------------------
# Score every clustering on the same scaled data. DBSCAN is scored twice because
# noise points belong to no cluster: once without them, once as one extra group
def silhouette_row(method_name, features, labels):
    return [method_name, len(set(labels)), len(labels), round(silhouette_score(features, labels), 3)]


print("=" * 70)
print("QUESTION 3b: SILHOUETTE SCORES")
print("=" * 70)

is_clustered = dbscan_labels != -1
score_rows = [
    silhouette_row(f"K-means (k = {K})", X_scaled, kmeans_labels),
    silhouette_row(f"Hierarchical (Ward, k = {K})", X_scaled, hierarchical_labels),
    silhouette_row("DBSCAN, noise excluded", X_scaled[is_clustered], dbscan_labels[is_clustered]),
    silhouette_row("DBSCAN, noise as own group", X_scaled, dbscan_labels),
]
print(
    tabulate(
        score_rows,
        headers=["Method", "Groups", "Customers scored", "Silhouette"],
        tablefmt="fancy_grid",
    )
)
