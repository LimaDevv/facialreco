import cv2

# Charger le classificateur de visages pré-entraîné
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Vérifier si le classificateur de visages est chargé avec succès
if face_cascade.empty():
    print("Erreur: Impossible de charger le classificateur de visages.")
    exit()

# Capturer la vidéo en direct à partir de la webcam
cap = cv2.VideoCapture(0)

while True:
    # Lire une image de la vidéo
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de capturer la vidéo.")
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Afficher l'image avec les visages détectés
    cv2.imshow('Face Detection', frame)

    # Attendre la touche 'q' pour quitter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture de la vidéo et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
