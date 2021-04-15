# Aalgoritmo Harris & Stephens

"""
 Vamos a buscar aquellas zonas con una variacion de imagenes mut grande. De esta forma obtendremos los puntos de interes.
 La idea del algoritmo es obtener aquellos puntos que hacen “esquina” mirando si tiene dos gradientes muy grandes en perpendicular.
"""

import cv2
import numpy as np
from datetime import datetime

camara = cv2.VideoCapture(2, cv2.CAP_DSHOW)
actualTime = 0
puntosRelevantes = 0
umbralSuperior = 1.5
umbralInferior = 0.5
now = 0
tiempoActual = 0

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

    #Obtenemos el numero total se esquinas
    nEsquinas = np.sum(harris > 0.01 * harris.max())

    #Cada 15 minutos se renuevan los puntos de referencia usados para comparar
    now = datetime.now()
    if now.timestamp() - tiempoActual > 9000000 or tiempoActual == 0:
        tiempoActual = now.timestamp()
        puntosRelevantes = nEsquinas

    # Para saber si se ha producido movimient lo que se hace es comparar si la cantidad de puntos entra dentro de un umbral.
    print("Numero de esquinas: " + str(nEsquinas))
    print("Umbral superior: " + str(puntosRelevantes * umbralSuperior))
    print("Umbral inferior: " + str(puntosRelevantes * umbralInferior))
    if nEsquinas > (puntosRelevantes * umbralSuperior) or nEsquinas < (puntosRelevantes * umbralInferior):
        print("MOVIMIENTO")

    # Del resultado anterior me quedo con el 1% de las esquinas totales más relevantes. Se represenán mediante puntos en azul.
    imagen[harris>0.01 * harris.max()] = [0, 0, 255]
    cv2.imshow("Harris - color", imagen);

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break