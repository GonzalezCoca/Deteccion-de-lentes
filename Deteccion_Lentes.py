import cv2
import numpy as np

# Cargar archivos de entrenamiento y configuracion YOLOV3
red = cv2.dnn.readNet("D:\VisionPorComputadora\Proyecto\yolov3_training_last (2).weights", "D:\VisionPorComputadora\Proyecto\yolov3_testing.cfg")
# Nombre del objeto a buscar
clases = ["Lentes"]

# cargar imagen
img = cv2.imread("D:\VisionPorComputadora\Proyecto\img32.jpg")
img = cv2.resize(img, None, fx=0.7, fy=0.7)
height, width, channels = img.shape

# Conectar con la red neuronal
blob = cv2.dnn.blobFromImage(img, 0.00784, (416, 416), (0, 0, 0), False, crop=False)
red.setInput(blob)
outs = red.forward(red.getUnconnectedOutLayersNames())

#Mostrar resultados
ids_clases = []
confidences = []
cajas = []

# Itera sobre las salidas de la red neuronal
for out in outs:
    for deteccion in out:
         # Obtiene las puntuaciones de confianza para las clases
        scores = deteccion[5:]
        # Obtiene el índice de la clase con la puntuación más alta
        clases_id = np.argmax(scores)
        # Obtiene la confianza (puntuación) de la clase predicha
        seguridad = scores[clases_id]
        if seguridad > 0.3:
            # Objecto detectado
            print(clases_id, seguridad)

            # Calcula las coordenadas y dimensiones del rectángulo delimitador
            centro_x = int(deteccion[0] * width)
            centro_y = int(deteccion[1] * height)
            w = int(deteccion[2] * width)
            h = int(deteccion[3] * height)

            # Coordenadas del rectangulo
            # Calcula las coordenadas (x, y) de la esquina superior izquierda del rectángulo
            x = int(centro_x - w / 2)
            y = int(centro_y - h / 2)

            # Almacena las coordenadas y dimensiones del rectángulo en la lista 'cajas'
            cajas.append([x, y, w, h])
            # Almacena la confianza en la lista 'confidences'
            confidences.append(float(seguridad))
            # Almacena el índice de la clase en la lista 'class_ids'
            ids_clases.append(clases_id)

# Evitar redundancia
# Realiza la supresión de no máximos en las detecciones
indices = cv2.dnn.NMSBoxes(cajas, confidences, 0.5, 0.4)
print(indices)

# Genera colores aleatorios para cada clase
colores = np.random.uniform(0, 255, size=(len(clases), 3))

# Define el tipo de fuente a utilizar para los textos
font = cv2.FONT_HERSHEY_COMPLEX

# Itera sobre las cajas delimitadoras
for i in range(len(cajas)):
    # Verifica si el índice i está presente en la lista de índices después de la supresión de no máximos
    if i in indices:
        # Obtiene las coordenadas (x, y, w, h) de la caja delimitadora actual
        x, y, w, h = cajas[i]
        # Obtiene la etiqueta de clase correspondiente al índice de clase ids_clases[i] y la convierte a cadena
        etiqueta = str(clases[ids_clases[i]])
        # Obtiene un color correspondiente a la clase de objeto utilizando el índice de clase ids_clases[i]
        color = colores[ids_clases[i]]
        # Dibuja un rectángulo alrededor de la detección en la imagen img
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
        # Muestra la etiqueta en la imagen img
        cv2.putText(img, etiqueta, (x, y - 8), font, 0.8, color, 3)

# Display images
cv2.imshow("Imagen con deteccion de Lentes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
