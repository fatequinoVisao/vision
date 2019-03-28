import cv2
import sys

# pegar imagem que deseja ler como parâmetro
caminho = sys.argv[1]
arquivoXMLrec = "C:\Users\matheus.bezerra\Downloads\Webcam-Face-Detect-master\webcam-face-detector\haarcascade_frontalface_default.xml" #inserir caminho do xml

# pegar a parada do xml
rostoRec = cv2.CascadeClassifier(arquivoXMLrec)

# ler a imagem
imagem = cv2.imread(caminho)
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# procurando rostos
rostos = rostoRec.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE //não vai precisar dessa fita
)

print("Foi encontrado {0} rosto(s)!".format(len(rostos)))

# desenhando o quadrado no rosto
for (x, y, w, h) in rostos:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Rostos capturados", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()