import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

# Definir una función para contar dedos

def countFingers(image, hand_landmarks, handNo=0):
    
    if hand_landmarks:
        # Obtener todos los puntos de referencia de la primera mano visible
        landmarks = hand_landmarks[handNo].landmark
        # print(landmarks)

        # Contar dedos
        fingers = []

        for lm_index in tipIds:
                # Obtener puntas de los dedos y valor de posición "y" inferior
                finger_tip_y = landmarks[lm_index].y 
                finger_bottom_y = landmarks[lm_index - 2].y

                # Verificar si algún dedo está abierto o cerrado
                if lm_index !=4:
                    if finger_tip_y < finger_bottom_y:
                        fingers.append(1)
                        print("El dedo con id ",lm_index," está abierto.")

                    if finger_tip_y > finger_bottom_y:
                        fingers.append(0)
                        print("El dedo con id ",lm_index," está cerrado.")

        # print(fingers)
        totalFingers = fingers.count(1)

        # Mostrar texto
        text = f'Fingers: {totalFingers}'

        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Definir una función para
def drawHandLanmarks(image, hand_landmarks):

    # Dibujar las conexiones entre los puntos de referencia
    if hand_landmarks:

      for landmarks in hand_landmarks:
               
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)


while True:
    success, image = cap.read()

    image = cv2.flip(image, 1)
    
    # Detectar los puntos de referencia de las manos
    results = hands.process(image)

    # Obtener la posición de los puntos de referencia del resultado procesado
    hand_landmarks = results.multi_hand_landmarks

    # Dibujar puntos de referencia
    drawHandLanmarks(image, hand_landmarks)

    # Obtener las posiciones de los dedos de la mano
    countFingers(image, hand_landmarks)

    cv2.imshow("Controlador de medios", image)

    # Cerrar la ventana al precionar la barra espaciadora
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()