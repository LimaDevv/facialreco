import cv2

# Charger les classificateurs pré-entraînés
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Vérifier si les classificateurs sont chargés avec succès
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("Erreur: Impossible de charger les classificateurs.")
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

        # Région d'intérêt pour les yeux et le sourire
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Détecter les yeux dans la région d'intérêt
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # Détecter les sourires dans la région d'intérêt
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    # Afficher le nombre de visages détectés
    cv2.putText(frame, f"Visages détectés: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher l'image avec les visages, yeux et sourires détectés
    cv2.imshow('Face Detection', frame)

    # Attendre la touche 'q' pour quitter la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture de la vidéo et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
