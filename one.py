import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import speech_recognition as sr

# Initialize text-to-speech
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Voice command processing
def voice_commands():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    with sr.Microphone() as source:
        speak("Voice control activated")
        while True:
            print("Listening...")
            try:
                audio = r.listen(source, timeout=5)
                command = r.recognize_google(audio).lower()
                print("You said:", command)
                if "hello" in command:
                    speak("Sat Shri Akaal")
                elif "clear" in command:
                    global canvas
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    speak("Canvas cleared")
                elif "exit" in command or "quit" in command:
                    speak("Closing application")
                    exit()
            except Exception as e:
                print("Could not recognize.")

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
success, img = cap.read()
height, width, _ = img.shape
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Start voice command thread
threading.Thread(target=voice_commands, daemon=True).start()

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append((cx, cy))

            if lmList:
                index_finger = lmList[8]
                middle_finger = lmList[12]
                pinky = lmList[20]

                # Drawing only with index finger up
                if abs(index_finger[1] - middle_finger[1]) > 40:
                    cv2.circle(img, index_finger, 15, (0, 255, 0), cv2.FILLED)
                    cv2.circle(canvas, index_finger, 15, (255, 255, 255), cv2.FILLED)

                # Clear canvas on open palm (distance between index finger and pinky close)
                distance = abs(index_finger[1] - pinky[1])
                if distance < 20:
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Drawing App", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
