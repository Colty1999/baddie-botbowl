import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Wczytaj dane z pliku CSV
df = pd.read_csv('game_results.csv')

# Wykres 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['Param1'], df['Param2'], df['Param3'], c=df['Wins'], cmap='viridis')
plt.colorbar(sc, label='Wins')
ax.set_xlabel('agresive')
ax.set_ylabel('safe_play')
ax.set_zlabel('prob_risk')
plt.title('3D Scatter Plot: Wins vs Parameters')
plt.show()


# Wykresy Liniowe
for fixed_param in ['Param1', 'Param2', 'Param3']:
    plt.figure(2)
    for value in df[fixed_param].unique():
        subset = df[df[fixed_param] == value]
        plt.plot(subset['Param1'], subset['Wins'], label=f'{fixed_param}={value}')
    plt.xlabel('agresive')
    plt.ylabel('Wins')
    plt.title(f'Line Plot: Wins vs Param1 with fixed {fixed_param}')
    plt.legend()
    plt.show()

# Wykresy Liniowe
for fixed_param in ['Param1', 'Param2', 'Param3']:
    plt.figure(3)
    for value in df[fixed_param].unique():
        subset = df[df[fixed_param] == value]
        plt.plot(subset['Param2'], subset['Wins'], label=f'{fixed_param}={value}')
    plt.xlabel('safe_play')
    plt.ylabel('Wins')
    plt.title(f'Line Plot: Wins vs Param2 with fixed {fixed_param}')
    plt.legend()
    plt.show()

# Wykresy Liniowe
for fixed_param in ['Param1', 'Param2', 'Param3']:
    plt.figure(4)
    for value in df[fixed_param].unique():
        subset = df[df[fixed_param] == value]
        plt.plot(subset['Param3'], subset['Wins'], label=f'{fixed_param}={value}')
    plt.xlabel('prob_risk')
    plt.ylabel('Wins')
    plt.title(f'Line Plot: Wins vs Param3 with fixed {fixed_param}')
    plt.legend()
    plt.show()
