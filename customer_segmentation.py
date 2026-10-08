import matplotlib

matplotlib.use("Agg")  # save plots to file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ---------------------------------------------------------------------------
# Step 1 / Question 2a: data preparation and choice of k
# ---------------------------------------------------------------------------
print("=" * 70)
print("QUESTION 2a: DATA PREPARATION AND CHOICE OF K")
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