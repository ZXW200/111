import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from Maping import COUNTRY_CODE

os.makedirs("CleanedData", exist_ok=True)
os.makedirs("CleanedDataPlt", exist_ok=True)

# 读取数据 Load data
df = pd.read_csv("CleanedData/cleaned_ictrp.csv", encoding="utf-8")

# 按国家聚合统计 Aggregate by country
country_data = []
for codes in df['country_codes'].dropna():
    for code in str(codes).upper().replace('|', ' ').split():
        if code in COUNTRY_CODE:
            country_data.append({
                'country': COUNTRY_CODE[code],
                'results_posted': df.loc[df['country_codes'].str.contains(code, na=False), 'results_posted'].iloc[0] if len(df[df['country_codes'].str.contains(code, na=False)]) > 0 else False,
                'sponsor_category': df.loc[df['country_codes'].str.contains(code, na=False), 'sponsor_category'].iloc[0] if len(df[df['country_codes'].str.contains(code, na=False)]) > 0 else 'Unknown'
            })

country_df = pd.DataFrame(country_data)

# 计算每个国家的特征 Calculate features for each country
country_features = country_df.groupby('country').agg(
    total_trials=('country', 'size'),
    publication_rate=('results_posted', 'mean'),
    industry_pct=('sponsor_category', lambda x: (x == 'Industry').mean()),
    government_pct=('sponsor_category', lambda x: (x == 'Government').mean())
).reset_index()

# 过滤试验数量>=5的国家 Filter countries with >=5 trials
country_features = country_features[country_features['total_trials'] >= 5]

# 特征标准化 Standardize features
features = ['total_trials', 'publication_rate', 'industry_pct', 'government_pct']
X = country_features[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means聚类 K-means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
country_features['cluster'] = kmeans.fit_predict(X_scaled)

# 保存结果 Save results
country_features.to_csv("CleanedData/country_clusters.csv", index=False, encoding="utf-8-sig")

# 可视化 Visualization
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i in range(3):
    cluster_data = country_features[country_features['cluster'] == i]
    ax.scatter(cluster_data['total_trials'],
               cluster_data['publication_rate']*100,
               c=colors[i], label=f'Cluster {i}', s=100, alpha=0.7, edgecolors='black')

ax.set_xlabel('Total Trials', fontsize=12, fontweight='bold')
ax.set_ylabel('Publication Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Country Clustering by Trial Characteristics', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("CleanedDataPlt/country_clustering.jpg", dpi=300, bbox_inches='tight')
plt.close()

# 输出聚类统计 Print cluster statistics
print("\n=== Country Clustering Results ===\n")
for i in range(3):
    cluster_data = country_features[country_features['cluster'] == i]
    print(f"Cluster {i}: {len(cluster_data)} countries")
    print(f"  Avg trials: {cluster_data['total_trials'].mean():.1f}")
    print(f"  Avg publication rate: {cluster_data['publication_rate'].mean()*100:.1f}%")
    print(f"  Avg industry %: {cluster_data['industry_pct'].mean()*100:.1f}%\n")

print(f"Results saved to CleanedData/country_clusters.csv")
print(f"Visualization saved to CleanedDataPlt/country_clustering.jpg")
