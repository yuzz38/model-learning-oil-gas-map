import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point, Polygon
import numpy as np
import random

# --- 1. Генерация тестовых данных с глубиной залегания ---
np.random.seed(42)
random.seed(42)

# Известные месторождения (10 точек) + глубина залегания (м)
known_deposits = np.array([
    # lat,   lon,    сейсмика, пористость (%), ГИС (Ом·м), глубина (м)
    [55.345, 50.210, 7.5,      18,             120,         1500],
    [55.230, 50.300, 6.8,      15,             150,         2300],
    [55.400, 50.400, 8.2,      22,              90,         1800],
    [55.500, 50.100, 5.5,      12,             200,         2800],
    [55.250, 50.450, 7.0,      20,             110,         1600],
    [55.350, 50.150, 6.5,      16,             140,         2100],
    [55.450, 50.350, 7.8,      14,             180,         2500],
    [55.300, 50.250, 6.0,      19,             100,         1400],
    [55.380, 50.280, 7.2,      17,             160,         1900],
    [55.420, 50.320, 8.0,      13,             190,         2700]
])

# Случайные точки в Татарстане (500 точек) + глубина
def generate_random_point():
    lat = np.random.uniform(54.9, 55.7)
    lon = np.random.uniform(49.8, 50.8)
    seismic = np.random.uniform(5.0, 9.0)
    porosity = np.random.uniform(10, 25)
    resistivity = np.random.uniform(80, 250)
    depth = np.random.uniform(1000, 3000)
    return [lat, lon, seismic, porosity, resistivity, depth]

potential_points = np.array([generate_random_point() for _ in range(500)])

# --- 2. Подготовка данных (включая глубину) ---
scaler = StandardScaler()
X_known = scaler.fit_transform(known_deposits[:, 2:])  # Все параметры, кроме координат
X_potential = scaler.transform(potential_points[:, 2:])

# --- 3. Кластеризация K-Means ---
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_known)
labels = kmeans.predict(X_potential)
potential_cluster_points = potential_points[labels == 0]  # Перспективные точки

# --- 4. Визуализация ---
fig, ax = plt.subplots(figsize=(14, 10))

# Границы Татарстана
tatarstan_polygon = Polygon([
    [49.8, 54.9], [49.8, 55.7], [50.8, 55.7], [50.8, 54.9], [49.8, 54.9]
])
tatarstan = gpd.GeoDataFrame(geometry=[tatarstan_polygon])
tatarstan.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.3)

# Известные месторождения (красные кресты + подписи с глубиной)
for i, point in enumerate(known_deposits):
    ax.scatter(
        point[1], point[0],
        s=100, c='red', marker='X',
        label='Известные месторождения' if i == 0 else ""
    )
    ax.text(
        point[1] + 0.01, point[0],
        f"{int(point[5])} м",
        fontsize=9, color='red'
    )

scatter = ax.scatter(
    potential_cluster_points[:, 1], potential_cluster_points[:, 0],
    s=potential_cluster_points[:, 3] * 3.5,  # Увеличил размер точек
    c=potential_cluster_points[:, 2],
    cmap='RdYlGn',
    vmin=5, vmax=8.5,  # Явно задал границы цветовой шкалы
    alpha=0.15 + 0.8*(potential_cluster_points[:, 5] - 1000) / 2000,  # Увеличил минимальную прозрачность
    edgecolors='k',  # Чёрная обводка для лучшей видимости
    linewidths=0.1,  # Тонкая граница
    label='Перспективные зоны'
)

# Усиленная цветовая шкала
cbar = plt.colorbar(scatter)
cbar.set_label('Сейсмическая активность (баллы)', fontsize=12)
cbar.ax.tick_params(labelsize=10)  # Увеличил размер подписей шкалы
# Легенда для глубины (кастомизация)
depth_legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markersize=10,
               markerfacecolor='green', alpha=0.9, label='Мелкие (1000-2000 м)'),
    plt.Line2D([0], [0], marker='o', color='w', markersize=10,
               markerfacecolor='green', alpha=0.3, label='Глубокие (2000-3000 м)')
]
ax.legend(handles=depth_legend_elements, loc='lower right')

plt.title("Перспективные зоны нефтегазовых месторождений (цвет: сейсмика, размер: пористость, прозрачность: глубина)",
          fontsize=12, pad=20)
plt.xlabel("Долгота", fontsize=12)
plt.ylabel("Широта", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
