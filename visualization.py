import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px

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

# 国家名称映射（ISO标准）Country name mapping (ISO standard)
country_mapping = {
    'United States': 'United States of America',
    'United Kingdom': 'United Kingdom',
    'DR Congo': 'Democratic Republic of the Congo',
    'Côte d\'Ivoire': 'Ivory Coast',
    'Timor-Leste': 'East Timor',
    'Bolivia': 'Bolivia',
    'Venezuela': 'Venezuela',
    'Tanzania': 'United Republic of Tanzania',
    'Laos': 'Lao PDR'
}

# 应用映射 Apply mapping
country_stats['country_mapped'] = country_stats['country'].replace(country_mapping)

# 创建热力图 Create heatmap
fig = px.choropleth(
    country_stats,
    locations='country_mapped',
    locationmode='country names',
    color='count',
    hover_name='country',
    hover_data={'country_mapped': False, 'count': True},
    color_continuous_scale='YlOrRd',
    labels={'count': 'Number of Experiments'},
    title='World Map: Number of NTD Clinical Trials by Country'
)

fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth'
    ),
    title={
        'text': 'World Map: Number of NTD Clinical Trials by Country',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'family': 'Arial', 'color': '#2c3e50'}
    },
    width=1400,
    height=700,
    coloraxis_colorbar=dict(
        title="Trial Count",
        thicknessmode="pixels",
        thickness=15,
        lenmode="pixels",
        len=300
    )
)

# 保存为HTML文件 Save as HTML file
fig.write_html('CleanedDataPlt/world_heatmap.html')
print("✓ World heatmap saved as CleanedDataPlt/world_heatmap.html")

# 尝试保存为静态图片（需要Chrome） Try to save as static image (requires Chrome)
try:
    fig.write_image('CleanedDataPlt/world_heatmap.png', width=1400, height=700)
    print("✓ World heatmap saved as CleanedDataPlt/world_heatmap.png")
except Exception as e:
    print("⚠ PNG export skipped (requires Chrome/Chromium installation)")
    print("  Interactive HTML version is available at CleanedDataPlt/world_heatmap.html")

print("\n所有可视化完成！ All visualizations completed!")
