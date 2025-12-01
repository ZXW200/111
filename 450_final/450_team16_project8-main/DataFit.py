import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

os.makedirs("CleanedDataPlt", exist_ok=True)

# 读取数据 Load data
df = pd.read_csv("CleanedData/cleaned_ictrp.csv")

# 准备特征和目标变量 Prepare features and target
features = ["phase", "study_type", "sponsor_category", "income_level"]
X = df[features]
y = df["results_posted"].astype(int)

# 划分训练测试集 Split train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 构建Pipeline：添加 class_weight='balanced' 处理类别不平衡
model = Pipeline([
    ("encoder", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features)
    ])),
    ("logit", LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
]).fit(X_train, y_train)

# 提取特征名和系数 Extract feature names and coefficients
feature_names = model.named_steps["encoder"].get_feature_names_out()
coefficients = model.named_steps["logit"].coef_[0]

# 构建结果表 Build results dataframe
results = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

# 保存结果 Save results
results.to_csv("CleanedData/logit_results.csv", index=False, encoding="utf-8-sig")

# 绘图 Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Logistic Regression Coefficients (Balanced Model)',
             fontsize=16, fontweight='bold')

# 按特征类型分组 Group by feature type
groups = {
    'Phase': 'cat__phase_',
    'Study Type': 'cat__study_type_',
    'Sponsor': 'cat__sponsor_category_',
    'Income Level': 'cat__income_level_'
}

for i, (title, prefix) in enumerate(groups.items()):
    ax = axes.flatten()[i]

    # 筛选该组特征 Filter features for this group
    group_data = results[results['feature'].str.startswith(prefix)].copy()

    if len(group_data) == 0:
        continue

    # 去掉前缀 Remove prefix
    group_data['short_name'] = group_data['feature'].str.replace(prefix, '', regex=False)
    group_data = group_data.sort_values('coefficient')

    # 绘制条形图 Draw bar chart
    colors = ['#d62728' if x < 0 else '#2ecc71' for x in group_data['coefficient']]
    ax.barh(group_data['short_name'], group_data['coefficient'],
            color=colors, alpha=0.75, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Coefficient', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # 添加图例 Add legend
    if i == 0:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.75, label='Positive'),
            Patch(facecolor='#d62728', alpha=0.75, label='Negative')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# 保存图片 Save plot
plt.tight_layout()
plt.savefig("CleanedDataPlt/coefficients_plot_balanced.jpg", dpi=300, bbox_inches='tight')
plt.close()

# ========== 模型评估 ==========
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(" Model Evaluation (Balanced)")

print(f"\nTrain Accuracy: {model.score(X_train, y_train):.4f}")
print(f" Test Accuracy: {model.score(X_test, y_test):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n Class Distribution:")
print(f"train: {y_train.value_counts().to_dict()}")
print(f"test: {y_test.value_counts().to_dict()}")

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  → predit 0 (No Results): {cm[:, 0].sum()}")
print(f"  → predit 1 (Results Posted): {cm[:, 1].sum()}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Results (0)', 'Results Posted (1)'], zero_division=0))

# ========== 聚类分析 Clustering Analysis ==========
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("\n" + "="*60)
print(" Clustering Analysis")
print("="*60)

# 使用编码后的特征进行聚类 Use encoded features for clustering
X_encoded = model.named_steps["encoder"].transform(X)

# 标准化特征 Standardize features
scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrix
X_scaled = scaler.fit_transform(X_encoded)

# 使用肘部法则找到最佳聚类数 Use elbow method to find optimal number of clusters
inertias = []
silhouette_scores = []
K_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘制肘部法则图 Plot elbow method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
ax1.set_title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score For Different k', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("CleanedDataPlt/clustering_elbow_method.jpg", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nElbow method plot saved: CleanedDataPlt/clustering_elbow_method.jpg")

# 选择最佳k值（基于轮廓系数） Choose optimal k based on silhouette score
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters (based on silhouette score): {optimal_k}")

# 使用最佳k值进行聚类 Perform clustering with optimal k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# 将聚类标签添加到原始数据 Add cluster labels to original data
df['cluster'] = cluster_labels

# 使用PCA降维到2D进行可视化 Use PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled.toarray() if hasattr(X_scaled, 'toarray') else X_scaled)

print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.4f}")

# 绘制聚类可视化 Plot cluster visualization
fig, ax = plt.subplots(figsize=(14, 10))

# 为每个聚类使用不同的颜色 Use different colors for each cluster
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

for i in range(optimal_k):
    mask = cluster_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
              c=[colors[i]], label=f'Cluster {i}',
              alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

# 绘制聚类中心 Plot cluster centers
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
          c='red', marker='X', s=300, edgecolors='black',
          linewidth=2, label='Centroids', zorder=10)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
ax.set_title('KMeans Clustering of Clinical Trials (PCA Projection)',
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("CleanedDataPlt/kmeans_clustering_pca.jpg", dpi=300, bbox_inches='tight')
plt.close()

print(f"Clustering visualization saved: CleanedDataPlt/kmeans_clustering_pca.jpg")

# 分析每个聚类的特征 Analyze characteristics of each cluster
print("\n" + "="*60)
print(" Cluster Characteristics Analysis")
print("="*60)

for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]
    print(f"\n Cluster {i} (n={len(cluster_data)}, {len(cluster_data)/len(df)*100:.1f}%)")
    print("-" * 50)

    # 各特征分布 Feature distribution
    for feature in features:
        print(f"\n  {feature}:")
        value_counts = cluster_data[feature].value_counts()
        for val, count in value_counts.head(3).items():
            print(f"    - {val}: {count} ({count/len(cluster_data)*100:.1f}%)")

    # 结果发布率 Results posting rate
    results_rate = cluster_data['results_posted'].mean()
    print(f"\n  Results Posted Rate: {results_rate:.2%}")

# 保存聚类结果 Save clustering results
cluster_summary = df.groupby('cluster').agg({
    'results_posted': ['count', 'mean'],
    'phase': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown',
    'study_type': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown',
    'sponsor_category': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown',
    'income_level': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
})

cluster_summary.columns = ['Trial_Count', 'Results_Posted_Rate', 'Most_Common_Phase',
                           'Most_Common_Study_Type', 'Most_Common_Sponsor', 'Most_Common_Income_Level']
cluster_summary = cluster_summary.sort_values('Trial_Count', ascending=False)

cluster_summary.to_csv("CleanedData/cluster_summary.csv", encoding="utf-8-sig")
print("\n✓ Cluster summary saved: CleanedData/cluster_summary.csv")

# 绘制聚类特征对比图 Plot cluster feature comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cluster Characteristics Comparison', fontsize=16, fontweight='bold')

# 1. 聚类大小 Cluster size
ax = axes[0, 0]
cluster_sizes = df['cluster'].value_counts().sort_index()
ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors, edgecolor='black', linewidth=1)
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('Number of Trials', fontsize=11)
ax.set_title('Cluster Size Distribution', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# 2. 结果发布率 Results posting rate by cluster
ax = axes[0, 1]
results_by_cluster = df.groupby('cluster')['results_posted'].mean().sort_index()
ax.bar(results_by_cluster.index, results_by_cluster.values, color=colors, edgecolor='black', linewidth=1)
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('Results Posted Rate', fontsize=11)
ax.set_title('Results Posting Rate by Cluster', fontweight='bold', fontsize=12)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# 3. 赞助商类别分布 Sponsor category distribution by cluster
ax = axes[1, 0]
sponsor_cluster = pd.crosstab(df['cluster'], df['sponsor_category'], normalize='index')
sponsor_cluster.plot(kind='bar', stacked=True, ax=ax,
                     color=['#3498db', '#2ecc71', '#e74c3c', '#95a5a6'])
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Sponsor Category by Cluster', fontweight='bold', fontsize=12)
ax.legend(title='Sponsor', fontsize=9, title_fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

# 4. 试验阶段分布 Phase distribution by cluster
ax = axes[1, 1]
phase_cluster = pd.crosstab(df['cluster'], df['phase'], normalize='index')
phase_cluster.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Trial Phase by Cluster', fontweight='bold', fontsize=12)
ax.legend(title='Phase', fontsize=8, title_fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("CleanedDataPlt/cluster_characteristics.jpg", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Cluster characteristics plot saved: CleanedDataPlt/cluster_characteristics.jpg")

print("\n" + "="*60)
print(" Clustering Analysis Completed!")
print("="*60)

