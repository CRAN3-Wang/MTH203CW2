import pandas as pd
import numpy as np
import ACO

df = pd.read_csv(r"examples/ch150.tsp.txt", sep=" ", skiprows=6, header=None, encoding='utf8')
node = list(df[0][0:-1])
num_points = len(node)
city_x = np.array(df[1][0:-1])
city_y = np.array(df[2][0:-1])
CityCoordinates = np.squeeze(np.dstack((city_x.T, city_y.T)))
print(CityCoordinates)