import dlib
import cv2

# carrega a imagem
img = cv2.imread('picture8.jpeg')

# detector de rosto
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# converte a imagem em escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectar rostos na imagem em escala de cinza
faces = face_detector.detectMultiScale(gray, 1.3, 5)

# desenha um retângulo em volta de cada rosto detectado
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)




# Carrega o detector de faces
detector = dlib.get_frontal_face_detector()

# Carrega o detector de pontos faciais
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Detecta as faces na imagem
faces = detector(gray)

# Para cada face detectada, encontra os pontos faciais e desenha um retângulo ao redor dos olhos
for face in faces:
    landmarks = predictor(gray, face)
    for n in range(36, 48):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)



#nariz

# detector de faces
detector = dlib.get_frontal_face_detector()
# detector de pontos de referência faciais
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# para cada face detectada, detecta o nariz e desenha um retângulo em volta dele
for face in faces:
    # detecta os pontos de referência faciais
    landmarks = predictor(gray, face)
    # obtém as coordenadas do nariz (ponto 30 no modelo 68 landmarks)
    nose = landmarks.part(30)
    x, y = nose.x, nose.y
    # define o tamanho do retângulo com base na distância entre os pontos 28 e 31 (no modelo 68 landmarks)
    left = landmarks.part(28)
    right = landmarks.part(31)
    width = (right.x - left.x) * 2
    height = width
    # desenha o retângulo em volta do nariz
    cv2.rectangle(img, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), (0, 255, 0), 2)




#boca 



# Detector de rosto
face_detector = dlib.get_frontal_face_detector()

# Detector de pontos de referência faciais
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Detecta rostos na imagem em escala de cinza
faces = face_detector(gray)

# Para cada rosto detectado, detecta a boca e desenha um retângulo em volta dela
for face in faces:
    # Detecta os pontos de referência faciais
    landmarks = predictor(gray, face)
    
    # Obtém as coordenadas dos pontos que definem a região da boca
    left = landmarks.part(48)
    right = landmarks.part(54)
    top = landmarks.part(51)
    bottom = landmarks.part(57)
    
    # Define o tamanho do retângulo em volta da boca
    width = right.x - left.x
    height = bottom.y - top.y
    
    # Desenha o retângulo em volta da boca
    cv2.rectangle(img, (left.x, top.y), (right.x, bottom.y), (0, 255, 0), 2)


# Redimensiona a imagem para que caiba na tela
height, width, channels = img.shape
max_width = 800
scale_factor = max_width / width
img_resized = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

# Exibe a imagem com os rostos e olhos detectados
cv2.imshow('Imagem', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()





               