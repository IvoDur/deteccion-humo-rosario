import cv2
import numpy as np
from database import Database

cap = cv2.VideoCapture("video_noche.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w, h)

ROJO = (0,0,255)
BLANCO = (255,255,255)

FUENTE = cv2.FONT_HERSHEY_PLAIN
ESCALA_FUENTE = 2
GROSOR_FUENTE = 2 #Pixeles
SEPARACION_TEXTO = w//2
SEPARACION_EJEY_TEXTO = ((h//10) // 5) * 4

desplazamiento = 0
VELOCIDAD_DESPLAZAMIENTO = 1

humo_detectado = False
HOLGURA_FRAMES = 5
frames_sin_deteccion = HOLGURA_FRAMES


valor_minimo = np.array([0,125,125])
valor_maximo = np.array([25,255,255])


def alerta(frame,desplazamiento):
    cv2.rectangle(frame, (0,0), (w,h//10), ROJO, -1)
    cv2.rectangle(frame, (0,h), (w,h-h//10), ROJO, -1)

    cv2.putText(frame, "ALERTA", (desplazamiento-w, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento-w+SEPARACION_TEXTO, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento+SEPARACION_TEXTO, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)

    cv2.putText(frame, "ALERTA", (desplazamiento-w, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento-w+SEPARACION_TEXTO, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento+SEPARACION_TEXTO, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)

while True:
  database = Database("fire_detections.db")
  ref, frame = cap.read()
  frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  valor_minimo = np.array([0,125,125])
  valor_maximo = np.array([25,255,255])

  mask = cv2.inRange(frame_hsv, valor_minimo, valor_maximo)
  res = cv2.bitwise_and(frame, frame, mask=mask)
  contornos, jerarquia = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if len(contornos) != 0:
    humo_detectado = True
    for contorno in contornos:
      if cv2.contourArea(contorno) > 500:
        x,y,width,height = cv2.boundingRect(contorno)
        cv2.rectangle(frame, (x,y), (x + width, y+height), (0,0,255), 3)
    database.insert(frame)

  if humo_detectado:
    alerta(frame, desplazamiento)
    frames_sin_deteccion = 0
  elif frames_sin_deteccion < HOLGURA_FRAMES:
    alerta(frame, desplazamiento)
    frames_sin_deteccion += 1

  cv2.imshow("Prueba", mask)
  cv2.imshow("frame", frame)



  tecla = cv2.waitKey(1)
  if tecla & 0xFF == ord("q"):
    break
  elif tecla & 0xFF == ord("h"):
        humo_detectado = not humo_detectado

  desplazamiento += VELOCIDAD_DESPLAZAMIENTO
  if desplazamiento > w:
      desplazamiento = 0

cap.release()
cv2.destroyAllWindows()

