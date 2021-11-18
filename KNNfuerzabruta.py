# Librerías requeridas

from sklearn.neighbors import KDTree
from scipy.spatial import distance
import pandas as pd
import numpy as np
import time

# Pruebas! 

#----------------------- KNN Usando Fuerza Bruta! -----------------------------#

# tiempo empezar 
start_time = time.time()

# Cargamos los datos
points = pd.read_csv("sample_data/test100.csv")
points = points.iloc[: , 1:]
points.to_numpy()


# Redondeo para adaptarse a la matriz en la pantalla
D = distance.squareform(distance.pdist(points))
# print(np.round(D, 1))  


# Realizamos argsortcada fila de la matriz de distancias para obtener para cada
# punto una lista de los puntos más cercanos:
closest = np.argsort(D, axis=1)
# print(closest)


# Nuevamente, vemos que cada punto está más cerca de sí mismo. Entonces,
# sin tener en cuenta eso, ahora podemos seleccionar los k puntos más cercanos:
# Para cada punto, encuentra los 3 puntos más cercanos.
k = 3 
# print(closest[:, 1:k+1])


# tiempo terminar
print("--- %s seconds ---" % (time.time() - start_time))