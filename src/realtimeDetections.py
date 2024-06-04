import torch
import cv2
import numpy as np
from pathlib import Path

# Nombres de las clases correspondientes a las detecciones
nombres_clases = ['cooling', 'pollen', 'varroa', 'wasps', 'health']

def detectar_objetos_en_tiempo_real(ruta_modelo, retraso_entre_fotogramas_ms=20):
    dispositivo = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    modelo = torch.hub.load('ultralytics/yolov5', 'custom', path=ruta_modelo)
    modelo.to(dispositivo)

    cap = cv2.VideoCapture(0)  # Usa la cámara USB (generalmente es el dispositivo 0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: No se pudo leer el fotograma de la cámara.")
            break

        results = modelo(frame)
        frame_con_resultados = np.squeeze(results.render())

        # Muestra las coincidencias por consola
        for detection in results.pred[0]:
            label = detection[-1].item()
            score = detection[4].item()
            print(f"Clase: {nombres_clases[int(label)]}, Confianza: {score}")

        cv2.imshow('Detector varroa', frame_con_resultados)

        # Agrega un retraso entre fotogramas
        cv2.waitKey(retraso_entre_fotogramas_ms)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Ruta del modelo

    directorio_actual = Path(__file__).resolve().parent
    ruta_modelo = directorio_actual.parent / 'model' / 'nuevo.pt'
    #ruta_modelo = 'C:/Users/ckill/Escritorio/Proyecto de titulo/Reconocer con yolov5/model/nuevo.pt'  # Reemplaza con la ruta correcta a tu modelo

    # Detectar objetos en tiempo real usando la cámara USB
    detectar_objetos_en_tiempo_real(ruta_modelo)
