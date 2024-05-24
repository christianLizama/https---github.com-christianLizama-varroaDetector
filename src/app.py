import torch
import cv2
from pathlib import Path
import numpy as np

# Nombres de las clases correspondientes a las detecciones
nombres_clases = ['cooling', 'pollen', 'varroa', 'wasps', 'health']

def detectar_objetos_en_video(ruta_modelo, ruta_video_entrada, ruta_video_salida, retraso_entre_fotogramas_ms=20):
    dispositivo = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    modelo = torch.hub.load('ultralytics/yolov5', 'custom', path=ruta_modelo)
    modelo.to(dispositivo)
    
    cap = cv2.VideoCapture(ruta_video_entrada)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ventana_ancho = 800  # Ancho de la ventana
    ventana_alto = int(frame_height * (ventana_ancho / frame_width))  # Calcula la altura proporcional

    out = cv2.VideoWriter(ruta_video_salida, cv2.VideoWriter_fourcc(*'XVID'), fps, (ventana_ancho, ventana_alto))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = modelo(frame)
        frame_con_resultados = np.squeeze(results.render())

        frame_con_resultados = cv2.resize(frame_con_resultados, (ventana_ancho, ventana_alto))

        out.write(frame_con_resultados)
        cv2.imshow('Detector varroa', frame_con_resultados)

        # Agrega un retraso entre fotogramas
        cv2.waitKey(retraso_entre_fotogramas_ms)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



def detectar_objetos_en_imagen(ruta_modelo, ruta_imagen):
    dispositivo = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    modelo = torch.hub.load('ultralytics/yolov5', 'custom', path=ruta_modelo)
    modelo.to(dispositivo)

    imagen = cv2.imread(ruta_imagen)

    # Redimensionar la imagen si no es de tamaño 640x640
    if imagen.shape[:2] != (160, 160):
        imagen = cv2.resize(imagen, (160, 160))

    results = modelo(imagen)
    results.print()
    imagen_con_resultados = np.squeeze(results.render())
    
    # Muestra las coincidencias por consola
    for detection in results.pred[0]:
        label = detection[-1].item()
        score = detection[4].item()
        print(f"Clase: {label}, Confianza: {score}")
    
    cv2.imshow('Detector varroa', imagen_con_resultados)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Rutas de los archivos y directorios
    directorio_actual = Path(__file__).resolve().parent
    ruta_modelo = directorio_actual.parent / 'model' / 'yolov5.pt'
    ruta_video_entrada = str(directorio_actual.parent / 'video' / 'video.avi')
    ruta_video_salida = str(directorio_actual.parent / 'video' / 'video_salida.avi')
    ruta_imagen = str(directorio_actual.parent / 'images' / 'varroa5.png')

    # Detectar objetos en un video
    #detectar_objetos_en_video(ruta_modelo, ruta_video_entrada, ruta_video_salida)

    # Detectar objetos en una imagen
    detectar_objetos_en_imagen(ruta_modelo, ruta_imagen)
