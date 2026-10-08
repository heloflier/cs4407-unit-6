# Unit 6: Customer Segmentation and Intelligent Promotions

A Python program that groups ten store customers into segments with three clustering algorithms (K-means, hierarchical clustering and DBSCAN) and compares them with silhouette scores. It supports the Unit 6 written assignment for CS 4407 Machine Learning.

## Contents

- `customer_segmentation.py`: the full program, in one script
- `data/customers.csv`: the ten-customer dataset (Customer ID, Age, Annual Spending, Purchases per Month)
- `requirements.txt`: the Python packages the script needs

## Workflow

The script runs top to bottom, and each section prints a banner so its output is easy to find:

1. **Question 2a, data preparation and choice of k:** drops Customer ID, standardizes the three features, then compares WCSS and silhouette for k = 1 to 6 and saves the elbow plot.
2. **Question 2b, K-means:** clusters the customers with k = 3, summarizes each cluster in the original units and saves a scatter plot.
3. **Question 2c, hierarchical clustering and DBSCAN:** builds a Ward-linkage tree cut into three clusters, saves the dendrogram, runs DBSCAN and prints a side-by-side comparison of the three methods.
4. **Question 3b, silhouette scores:** scores each method on the same scaled data. DBSCAN is scored twice, once without its noise points and once with them as their own group.

## Running

From the project folder, with a virtual environment active:

```
pip install -r requirements.txt
python customer_segmentation.py
```

The script reads `data/customers.csv` by path, so run it from the project folder. It prints the tables to the terminal and saves three images there: `elbow_method.png`, `kmeans_clusters.png` and `dendrogram.png`.
