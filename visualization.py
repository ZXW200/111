import pandas as pd
import matplotlib.pyplot as plt
import os
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

# 创建世界地图热力图 Create world map heatmap using world_map_4651_r2_mar25.jpg
print("\n生成世界地图热力图... Generating world map heatmap...")
country_stats = pd.read_csv("CleanedData/country_statistics.csv", encoding="utf-8-sig")

# 读取底图 Read base map
from PIL import Image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

base_map = Image.open('world_map_4651_r2_mar25.jpg')
img_width, img_height = base_map.size

# 国家中心点坐标 (经度, 纬度) Country center coordinates (longitude, latitude)
country_coords = {
    'Brazil': (-55, -10),
    'India': (78, 20),
    'Argentina': (-64, -34),
    'Kenya': (37, 1),
    'Tanzania': (35, -6),
    'Ethiopia': (39, 8),
    "Côte d'Ivoire": (-5, 8),
    'Uganda': (32, 1),
    'Spain': (-4, 40),
    'United States': (-95, 38),
    'Bangladesh': (90, 24),
    'Sudan': (30, 15),
    'China': (105, 35),
    'Bolivia': (-65, -17),
    'Colombia': (-74, 4),
    'Senegal': (-14, 14),
    'Netherlands': (5, 52),
    'United Kingdom': (-2, 54),
    'Laos': (102, 18),
    'Mali': (-4, 17),
    'Myanmar': (96, 22),
    'Ghana': (-2, 8),
    'Benin': (2, 9),
    'Peru': (-75, -10),
    'Burkina Faso': (-2, 13),
    'Cameroon': (12, 6),
    'Nigeria': (8, 9),
    'Democratic Republic of the Congo': (25, -4),
    'Zambia': (28, -15),
    'Mozambique': (35, -18),
    'South Africa': (25, -29),
    'France': (2, 47),
    'Germany': (10, 51),
    'Italy': (12, 42),
    'Mexico': (-102, 23),
    'Thailand': (101, 15),
    'Indonesia': (113, -2),
    'Australia': (133, -27),
    'Canada': (-95, 60),
    'Madagascar': (47, -19),
    'Morocco': (-7, 32),
    'Egypt': (30, 27),
    'Saudi Arabia': (45, 24),
    'Iran': (53, 32),
    'Pakistan': (69, 30),
    'Afghanistan': (67, 33),
    'Nepal': (84, 28),
    'Vietnam': (108, 16),
    'Philippines': (122, 13),
    'Japan': (138, 36),
    'South Korea': (128, 37),
    'Turkey': (35, 39),
    'Greece': (22, 39),
    'Poland': (19, 52),
    'Sweden': (15, 62),
    'Norway': (10, 60),
    'Belgium': (4, 51),
    'Switzerland': (8, 47),
    'Austria': (14, 47),
    'Czech Republic': (15, 50),
    'Portugal': (-8, 39),
}

# 将经纬度转换为图片坐标 Convert lat/lon to image coordinates
def latlon_to_pixels(lon, lat, img_width, img_height):
    # 标准地图投影: 经度 [-180, 180] -> [0, width], 纬度 [90, -90] -> [0, height]
    x = (lon + 180) * (img_width / 360)
    y = (90 - lat) * (img_height / 180)
    return x, y

# 准备数据 Prepare data
coords_x = []
coords_y = []
sizes = []
colors_data = []

for _, row in country_stats.iterrows():
    country = row['country']
    count = row['count']
    if country in country_coords:
        lon, lat = country_coords[country]
        x, y = latlon_to_pixels(lon, lat, img_width, img_height)
        coords_x.append(x)
        coords_y.append(y)
        sizes.append(count * 50)  # 点的大小与试验数量成正比
        colors_data.append(count)

# 创建图表 Create figure
fig, ax = plt.subplots(figsize=(20, 10))

# 显示底图 Display base map
ax.imshow(base_map, extent=[0, img_width, img_height, 0])

# 绘制热力图散点 Plot heatmap scatter points
if coords_x:
    scatter = ax.scatter(coords_x, coords_y, s=sizes, c=colors_data,
                        cmap='YlOrRd', alpha=0.6, edgecolors='darkred', linewidth=1)

    # 添加颜色条 Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Number of NTD Clinical Trials', fontsize=12)

ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)
ax.axis('off')

# 设置标题 Set title
ax.set_title('World Map: Number of NTD Clinical Trials by Country',
             fontsize=16, fontweight='bold', pad=20)

# 保存图片 Save image
plt.savefig('CleanedDataPlt/world_heatmap.jpg', dpi=300, bbox_inches='tight')
plt.close()
print("✓ World heatmap saved as CleanedDataPlt/world_heatmap.jpg")
print("\n所有可视化完成！ All visualizations completed!")
