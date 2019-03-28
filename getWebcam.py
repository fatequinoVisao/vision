import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

arquivoXMLrec = "haarcascade_frontalface_default.xml"
rostoRec = cv2.CascadeClassifier(arquivoXMLrec)
log.basicConfig(filename='webcam.log',level=log.INFO)

capturandoVideo = cv2.VideoCapture(0)

anterior = 0
while True:
    if not capturandoVideo.isOpened():
        print('Arruma essa camera ai!')
        sleep(5)
        pass

    # Captura frame por frame
    ret, frame = capturandoVideo.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostos = rostoRec.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Desenha um retangulo em volta dos rostos
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(rostos):
        anterior = len(rostos)
        log.info("rostos: "+str(len(rostos))+" data: "+str(dt.datetime.now()))


    # Mostra o frame com o retangulo
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam (nao ha dispose automatico, porra de python)
capturandoVideo.release()
cv2.destroyAllWindows()
