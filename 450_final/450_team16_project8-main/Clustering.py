import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import re

os.makedirs("CleanedData", exist_ok=True)
os.makedirs("CleanedDataPlt", exist_ok=True)

# 读取数据 Load data
df = pd.read_csv("CleanedData/cleaned_ictrp.csv", encoding="utf-8")

# 提取儿童参与特征 Extract children inclusion
def includes_children(age_min):
    if pd.isna(age_min):
        return 0
    age_str = str(age_min).lower()
    # 提取数字 Extract number
    match = re.search(r'(\d+)', age_str)
    if not match:
        return 0
    age_num = int(match.group(1))
    # 判断是否包含儿童 (<18岁) Check if includes children (<18 years)
    if 'month' in age_str or 'm' == age_str[-1]:
        return 1  # 按月算，肯定是儿童 Months = children
    elif 'year' in age_str or 'y' in age_str:
        return 1 if age_num < 18 else 0
    return 0

# 提取孕妇参与特征 Extract pregnant inclusion
def includes_pregnant(preg):
    if pd.isna(preg):
        return 0
    return 1 if str(preg).upper().strip() == 'INCLUDED' else 0

# 特征工程 Feature engineering
df['children_included'] = df['inclusion_age_min'].apply(includes_children)
df['pregnant_included'] = df['pregnant_participants'].apply(includes_pregnant)

# 编码疾病类型 Encode disease type
disease_dummies = pd.get_dummies(df['standardised_condition'], prefix='disease')

# 编码研究阶段 Encode phase
phase_dummies = pd.get_dummies(df['phase'], prefix='phase')

# 合并特征 Combine features
features_df = pd.concat([
    df[['children_included', 'pregnant_included']],
    disease_dummies,
    phase_dummies
], axis=1)

# 标准化 Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

# K-means聚类 (k=3) K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 保存结果 Save results
result_df = df[['trial_id', 'standardised_condition', 'phase',
                'children_included', 'pregnant_included', 'cluster']]
result_df.to_csv("CleanedData/trial_clusters.csv", index=False, encoding="utf-8-sig")

# 分析每个簇的特征 Analyze each cluster
print("\n=== Trial Clustering Results (RQ 2.2.4) ===\n")
for i in range(3):
    cluster_df = df[df['cluster'] == i]
    print(f"Cluster {i}: {len(cluster_df)} trials")
    print(f"  Children included: {cluster_df['children_included'].mean()*100:.1f}%")
    print(f"  Pregnant included: {cluster_df['pregnant_included'].mean()*100:.1f}%")
    print(f"  Top diseases: {cluster_df['standardised_condition'].value_counts().head(2).to_dict()}")
    print(f"  Top phases: {cluster_df['phase'].value_counts().head(2).to_dict()}\n")

# 可视化 Visualization
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71']

for i in range(3):
    cluster_df = df[df['cluster'] == i]
    ax.scatter(cluster_df['children_included'] + np.random.normal(0, 0.05, len(cluster_df)),
               cluster_df['pregnant_included'] + np.random.normal(0, 0.05, len(cluster_df)),
               c=colors[i], label=f'Cluster {i} (n={len(cluster_df)})',
               s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Children Included', fontsize=12, fontweight='bold')
ax.set_ylabel('Pregnant Women Included', fontsize=12, fontweight='bold')
ax.set_title('Trial Clustering: Vulnerable Population Inclusion Patterns',
             fontsize=13, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['No', 'Yes'])
ax.set_yticklabels(['No', 'Yes'])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("CleanedDataPlt/trial_clustering.jpg", dpi=300, bbox_inches='tight')
plt.close()

print("Results saved to CleanedData/trial_clusters.csv")
print("Visualization saved to CleanedDataPlt/trial_clustering.jpg")
