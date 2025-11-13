import pandas as pd
import matplotlib.pyplot as plt
import os
import folium
from CleanData import COUNTRY_CODE

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

# 反转映射：国家名 -> ISO代码 Reverse mapping: country name -> ISO code
name_to_code = {v: k for k, v in COUNTRY_CODE.items()}
country_stats['iso_alpha'] = country_stats['country'].map(name_to_code)

# 创建folium地图 Create folium map
m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles='CartoDB positron',
    prefer_canvas=True
)

# 添加choropleth层 Add choropleth layer
choropleth = folium.Choropleth(
    geo_data='CleanedData/geo_data/world-countries.json',
    data=country_stats,
    columns=['iso_alpha', 'count'],
    key_on='feature.id',
    fill_color='YlOrRd',
    fill_opacity=0.8,
    line_opacity=0.5,
    legend_name='Number of NTD Clinical Trials',
    nan_fill_color='#eeeeee',
    nan_fill_opacity=0.3,
    highlight=True
)
choropleth.add_to(m)

# 添加tooltip显示国家信息 Add tooltips
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(
        fields=['name'],
        aliases=['Country:'],
        style='background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;'
    )
)

# 为每个有数据的国家添加详细信息
import json
with open('CleanedData/geo_data/world-countries.json', 'r') as f:
    geo_json = json.load(f)

# 创建国家数据字典
country_data = dict(zip(country_stats['iso_alpha'], country_stats['count']))

# 更新GeoJSON添加count信息（所有国家都添加，没有数据的设为"No data"）
for feature in geo_json['features']:
    country_id = feature.get('id')
    if country_id in country_data:
        feature['properties']['trials'] = f"{int(country_data[country_id])} trials"
    else:
        feature['properties']['trials'] = 'No data'

# 添加带有详细信息的GeoJson层
style_function = lambda x: {
    'fillColor': 'transparent',
    'color': 'transparent',
    'weight': 0,
    'fillOpacity': 0
}

highlight_function = lambda x: {
    'weight': 3,
    'color': '#ff6600',
    'fillOpacity': 0.7
}

folium.GeoJson(
    geo_json,
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=folium.GeoJsonTooltip(
        fields=['name', 'trials'],
        aliases=['Country:', 'Clinical Trials:'],
        style='background-color: white; color: #333333; font-family: arial; font-size: 14px; padding: 10px;',
        sticky=False
    )
).add_to(m)

# 添加标题
title_html = '''
<div style="position: fixed;
     top: 10px; left: 50px; width: 550px; height: auto;
     background-color: white; border: 2px solid grey; z-index: 9999;
     font-size: 16px; font-weight: bold; padding: 15px;
     box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
     <span style="color: #2c3e50;">🌍 World Map: Number of NTD Clinical Trials by Country</span>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 保存HTML Save HTML
m.save('CleanedDataPlt/world_heatmap.html')
print("✓ World heatmap saved as CleanedDataPlt/world_heatmap.html")
print("  Hover over countries to see trial counts")
print("\n所有可视化完成！ All visualizations completed!")
