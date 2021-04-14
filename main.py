# Aalgoritmo Harris & Stephens

"""
 Vamos a buscar aquellas zonas con una variacion de imagenes mut grande. De esta forma obtendremos los puntos de interes.
 La idea del algoritmo es obtener aquellos puntos que hacen “esquina” mirando si tiene dos gradientes muy grandes en perpendicular.
"""

import cv2
import numpy as np

camara = cv2.VideoCapture(1)

while True:
    _, imagen = camara.read()

    # La imágnen a color se trnsforma en una imágen en gris
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Se transforma la imagen en gri en un float
    imagenGris = np.float32(imagenGris)

    """
     Se aplica la fórmula con los siguientes parámetros:
     
        1- Imágen sobre la que buscar los puntos.
        
        2- Tamaño de los vecinos considerados para la detección, es decir, se utiliza el rango de valores -2 < x < 2.
        
        3- Parámetro de apertura de la derivada Sobel, es decir, M e una matriz 3x3. Cuanto más grande más falsos positivos. 
           Cuanto más pequeño más puntos de interés se pierden.
           
        4- Parámetro libre en la formula de Harris. Es una constante que varía entre [0.04, 0.06].
        
                                                                                    -1  0  1    -1 -2 -1
        Las matrizes que se aplicarán para buscar las esquinas son las siguientes:  -2  0  2     0  0  0
                                                                                    -1  0  1     1  2  1
    """
    harris = cv2.cornerHarris(imagenGris, 2, 3, 0.04)

    # Del resultado anterior me quedo con el 1% de las esquinas totales más relevantes. Se represenán mediante puntos en azul.
    imagen[harris>0.01 * harris.max()] = [255, 0, 0]

    cv2.imshow("Harris", imagen);
    cv2.waitKey(0)