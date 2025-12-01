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

# 使用编码后的特征进行聚类 Use encoded features for clustering
X_encoded = model.named_steps["encoder"].transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_encoded)

# 统计各簇的发表率 Calculate publication rate by cluster
df['cluster'] = clusters
cluster_stats = df.groupby('cluster').agg({
    'results_posted': ['count', 'sum', 'mean']
}).round(3)

print("\n\n=== Cluster Analysis ===")
print("Trials grouped by characteristics:\n")
for i in range(3):
    total = cluster_stats.loc[i, ('results_posted', 'count')]
    published = cluster_stats.loc[i, ('results_posted', 'sum')]
    rate = cluster_stats.loc[i, ('results_posted', 'mean')]
    print(f"Cluster {i}: {int(total)} trials, {int(published)} published ({rate*100:.1f}%)")

# 可视化 Visualization
fig, ax = plt.subplots(figsize=(10, 6))
cluster_ids = [0, 1, 2]
counts = [cluster_stats.loc[i, ('results_posted', 'count')] for i in cluster_ids]
rates = [cluster_stats.loc[i, ('results_posted', 'mean')] * 100 for i in cluster_ids]

x = np.arange(len(cluster_ids))
width = 0.35

ax2 = ax.twinx()
bars1 = ax.bar(x - width/2, counts, width, label='Total Trials', color='#3498db', alpha=0.7)
bars2 = ax2.bar(x + width/2, rates, width, label='Publication Rate (%)', color='#e74c3c', alpha=0.7)

ax.set_xlabel('Cluster', fontweight='bold')
ax.set_ylabel('Number of Trials', fontweight='bold')
ax2.set_ylabel('Publication Rate (%)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
ax.set_title('Trial Clustering by Publication Patterns', fontweight='bold', pad=20)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig("CleanedDataPlt/cluster_publication.jpg", dpi=300, bbox_inches='tight')
plt.close()

# KMeans聚类可视化 KMeans clustering visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_encoded)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#3498db', '#e74c3c', '#2ecc71']
published_all = df['results_posted'].values

for i in range(3):
    mask = clusters == i

    # 已发表的用实心，未发表用空心 Published=filled, unpublished=hollow
    ax.scatter(X_pca[mask & published_all, 0], X_pca[mask & published_all, 1],
               c=colors[i], label=f'Cluster {i}', s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.scatter(X_pca[mask & ~published_all, 0], X_pca[mask & ~published_all, 1],
               c=colors[i], s=60, alpha=0.3, edgecolors=colors[i], linewidths=1.5, facecolors='none')

# 画聚类中心 Plot cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=200, alpha=0.8,
           marker='X', edgecolors='white', linewidths=2, label='Centers')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax.set_title('KMeans Clustering (PCA Projection)', fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("CleanedDataPlt/kmeans_clusters.jpg", dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved: CleanedDataPlt/cluster_publication.jpg")
print("KMeans plot saved: CleanedDataPlt/kmeans_clusters.jpg")