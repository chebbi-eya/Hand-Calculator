import cv2
import mediapipe as mp
import sys

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands 

hands = mpHands.Hands(max_num_hands=2)

mpDraw = mp.solutions.drawing_utils 
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    totalFingers = 0  


    if results.multi_hand_landmarks:
        for hand_index, handLms in enumerate(results.multi_hand_landmarks):
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape 
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy]) 
 
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 
 
                if id == 8:
                    cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)

            if len(lmList) == 21:
                fingers = []

                if hand_index == 0:  
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:  # Main gauche
                    if lmList[tipIds[0]][1] < lmList[tipIds[0] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Autres doigts
                for tip in range(1, 5):
                    if lmList[tipIds[tip]][2] < lmList[tipIds[tip] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers += fingers.count(1)

        cv2.putText(img, f'Doigts: {totalFingers}', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

    cv2.imshow('hand tracker', img)

    if cv2.waitKey(5) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
