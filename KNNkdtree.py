# Librerías requeridas

from sklearn.neighbors import KDTree
from scipy.spatial import distance
import pandas as pd
import numpy as np
import time

# Pruebas! 

#----------------------- Usando KNN con KDTree! --------------------------#

# tiempo empezar 
start_time = time.time()

# Cargamos los datos
points = pd.read_csv("sample_data/test100.csv")
points = points.iloc[: , 1:]
points.to_numpy()

# Calculamos las distancias con la metrica euclediana
kdt = KDTree(points, leaf_size=2, metric='euclidean')

# Imprimimos el resulado
kdt.query(points, k=3, return_distance=True)

# tiempo terminar
print("--- %s seconds ---" % (time.time() - start_time))