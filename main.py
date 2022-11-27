from time import sleep
import datetime
from subprocess import Popen
from sys import executable

print(datetime.datetime.now().hour)

noche_ejecutado = False
dia_ejecutado = False
extProc = None
#executable almacena la ubicacion del interprete de python

while True:

    if datetime.datetime.now().hour >= 20 and not noche_ejecutado:
        try:
            Popen.terminate(extProc)
        except:
            pass
        dia_ejecutado = False
        extProc = Popen([executable, "deteccion_noche.py"])
        noche_ejecutado = True
    elif datetime.datetime.now().hour < 20 and not dia_ejecutado:
        try:
            Popen.terminate(extProc)
        except:
            pass
        noche_ejecutado = False
        extProc = Popen([executable, "ejemplo_video.py"])
        dia_ejecutado = True

    sleep(300)
    print("Se intento cambiar el programa")
