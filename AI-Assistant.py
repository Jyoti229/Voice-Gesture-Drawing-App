import cv2
import mediapipe as mp
import pyttsx3
import speech_recognition as sr
import threading

# ========== Text-to-Speech Engine ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

# ========== Speech Recognition ==========
recognizer = sr.Recognizer()

def listen_and_respond():
    with sr.Microphone() as source:
        speak("Say something...")
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='en-IN')
            print("You said:", text)

            # Responses
            if "hello" in text.lower():
                speak("Hello! How can I help you?")
            elif "Sat sri akaal" in text.lower() or "sat shri akaal" in text.lower():
                speak("Sat Sri Akaal Ji!")
            elif "draw" in text.lower():
                speak("Starting hand drawing mode")
                start_drawing()
            else:
                speak("Sorry, I didn't understand.")
        except sr.UnknownValueError:
            speak("Sorry, couldn't recognize.")
        except sr.RequestError:
            speak("Speech service error.")

# ========== Hand Gesture Drawing ==========
def start_drawing():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    canvas = None

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                h, w, _ = frame.shape
                x, y = int(lm[8].x * w), int(lm[8].y * h)  # Index finger tip
                cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        cv2.imshow("Hand Drawing - Press 'q' to Exit", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = None

    cap.release()
    cv2.destroyAllWindows()

# ========== Main Thread ==========
if __name__ == "__main__":
    speak("Welcome! Say 'draw' to start drawing with hand.")
    while True:
        listen_and_respond()
