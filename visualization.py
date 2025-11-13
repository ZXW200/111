import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import pycountry
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# 创建输出文件夹 Create output folder
os.makedirs("CleanedDataPlt", exist_ok=True)

# 读取数据 Read Data
df = pd.read_csv("CleanedData/cleaned_ictrp.csv", encoding="utf-8")
print(f"Total trials: {len(df)}")

# 筛选已发表的试验 Selected published 
published_df = df[df["results_posted"] == True]
print(f"Published: {len(published_df)}")
print(f"Unpublished: {len(df) - len(published_df)}\n")

# 统计赞助商类别  Statistics on sponsor categories
all_sponsor_counts = df["sponsor_category"].value_counts()
published_sponsor_counts = published_df["sponsor_category"].value_counts()

# 指定类别颜色 Specify category color
color_map = {
    'Industry': '#3498db',
    'Non-profit': '#e74c3c',
    'Government': '#2ecc71',
    'Other': '#95a5a6'
}

# 绘制对比图 Draw a comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for data, ax, title in [(all_sponsor_counts, ax1, 'All Trials'),
                        (published_sponsor_counts, ax2, 'Published Trials')]:

    # 根据类别获取对应颜色 Obtain corresponding colors based on categories
    colors = [color_map[cat] for cat in data.index]

    wedges, texts, autotexts = ax.pie(
        data.values,
        labels=data.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )

    for autotext in autotexts:
        autotext.set_color('white')

    labels = [f'{cat}: {count}' for cat, count in zip(data.index, data.values)]
    ax.legend(labels, loc='upper left', fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold')

fig.suptitle('Sponsor Category Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('CleanedDataPlt/sponsor_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

industry_stats = pd.read_csv("CleanedData/country_Industry.csv", encoding="utf-8-sig")

# 定义高负担国家列表 Define the list of high burden countries
high_burden_countries = ['India', 'Mexico', 'Tanzania', 'Bangladesh', 'Bolivia',
                       'Côte d\'Ivoire', 'Kenya', 'Egypt']

# 添加负担分类列 Add burden classification column
industry_stats['burden_level'] = industry_stats['country'].apply(
    lambda x: 'High Burden' if x in high_burden_countries else 'Normal'
)

# 保存更新后的文件 Save
industry_stats.to_csv("CleanedData/country_Industry_HighBurden.csv",
                      index=False, encoding="utf-8-sig")
burden_sum = industry_stats.groupby('burden_level')['count'].sum() # count burden level

# 画图 plt
fig, ax = plt.subplots(figsize=(10, 7))
ax.pie(burden_sum.values, labels=burden_sum.index, autopct='%1.1f%%',
       colors=['#e74c3c', '#3498db'], startangle=90)
ax.set_title('Industry Trials by Region', fontsize=14, fontweight='bold')
plt.savefig('CleanedDataPlt/industry_region.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure saved successfully!")

# 创建世界地图热力图 Create world map heatmap
print("\n生成世界地图热力图... Generating world map heatmap...")
country_stats = pd.read_csv("CleanedData/country_statistics.csv", encoding="utf-8-sig")

# 读取世界地图GeoJSON Load world map GeoJSON from local directory
world = gpd.read_file('CleanedData/geo_data/world-countries.json')

# 国家名称到ISO-3代码的转换 Convert country names to ISO-3 codes
def get_iso3(country_name):
    """Get ISO-3 country code from country name"""
    # 手动映射特殊情况 Manual mapping for special cases
    special_cases = {
        'United States': 'USA',
        'United Kingdom': 'GBR',
        'DR Congo': 'COD',
        'Côte d\'Ivoire': 'CIV',
        'Timor-Leste': 'TLS',
        'Bolivia': 'BOL',
        'Venezuela': 'VEN',
        'Tanzania': 'TZA',
        'Laos': 'LAO',
        'Vietnam': 'VNM',
        'South Africa': 'ZAF'
    }

    if country_name in special_cases:
        return special_cases[country_name]

    try:
        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
    except:
        return None

# 添加ISO-3代码 Add ISO-3 codes
country_stats['iso_alpha'] = country_stats['country'].apply(get_iso3)

# 合并数据 Merge data with world map
world_merged = world.merge(country_stats, left_on='id', right_on='iso_alpha', how='left')

# 创建图形 Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 10))

# 创建颜色映射 Create colormap
colors = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('YlOrRd', colors, N=n_bins)

# 绘制热力图 Plot heatmap
world_merged.plot(
    column='count',
    ax=ax,
    legend=True,
    cmap=cmap,
    edgecolor='#333333',
    linewidth=0.3,
    missing_kwds={
        'color': '#e8e8e8',
        'edgecolor': '#999999',
        'linewidth': 0.3
    },
    legend_kwds={
        'label': 'Number of Clinical Trials',
        'orientation': 'horizontal',
        'shrink': 0.6,
        'aspect': 30,
        'pad': 0.05
    },
    vmin=0,
    vmax=country_stats['count'].max()
)

# 设置标题和样式 Set title and style
ax.set_title('World Map: Number of NTD Clinical Trials by Country',
             fontsize=22, fontweight='bold', pad=20, color='#2c3e50')
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
ax.axis('off')

# 添加注释 Add note
ax.text(0.02, 0.02, f'Total countries: {len(country_stats)} | Maximum trials: {country_stats["count"].max()} (Brazil)',
        transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 保存为PNG Save as PNG
plt.tight_layout()
plt.savefig('CleanedDataPlt/world_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ World heatmap saved as CleanedDataPlt/world_heatmap.png")
print("\n所有可视化完成！ All visualizations completed!")
