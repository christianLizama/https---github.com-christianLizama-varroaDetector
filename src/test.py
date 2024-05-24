import torch
import cv2
from PIL import Image

# Cargar el modelo pre-entrenado
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ckill/Escritorio/Proyecto de titulo/Reconocer con yolov5/model/yolov5.pt')

# Ruta de la imagen
image_path = 'C:/Users/ckill/Escritorio/Proyecto de titulo/Reconocer con yolov5/images/varroa2.jpg'

# Cargar la imagen usando OpenCV
img = cv2.imread(image_path)

# Redimensionar la imagen a 640x640
new_size = (640, 640)
img = cv2.resize(img, new_size)

# Convertir la imagen de BGR a RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Realizar la predicción en la imagen redimensionada
results = model(img)

# Mostrar los resultados de detección en la imagen
results.show()