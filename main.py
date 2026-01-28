import numpy as np
import pandas as pd

data = {
    'Game': ['Cyberpunk 2077', 'Elden Ring', 'Minecraft'],
    'Rating': [9.0, 9.5, 10.0],
    'Price_USD': [60, 60, 30],
    'Played_Hours': [120, 200, 5000]
}
df = pd.DataFrame(data)
df['Price_per_Hour'] = df['Price_USD'] / df['Played_Hours']
df = df.sort_values(by='Price_per_Hour', ascending=True)
df.to_csv('my_games_analytics.csv', index=False)
print(df)