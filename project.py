import cv2
import tkinter as tk
from tkinter import ttk
import threading
import speech_recognition as sr
import pyttsx3
import mediapipe as mp

# === Text-to-Speech Engine Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_in_language(text):
    engine.say(text)
    engine.runAndWait()

# === Drawing Flag ===
drawing_enabled = False

# === Language Setting ===
language = "en-IN"

# === GUI Setup ===
root = tk.Tk()
root.title("Voice + Gesture Drawing App")
root.geometry("600x400")

text_box = tk.Text(root, height=15, width=70)
text_box.pack(pady=10)

lang_var = tk.StringVar(value=language)
lang_dropdown = ttk.Combobox(root, textvariable=lang_var, values=["en-IN", "hi-IN", "pa-IN"])
lang_dropdown.pack()

def update_language(event):
    global language
    language = lang_var.get()
    text_box.insert(tk.END, f"Language manually set to: {language}\n")
    text_box.see(tk.END)

lang_dropdown.bind("<<ComboboxSelected>>", update_language)

def listen_for_commands():
    global language, drawing_enabled
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        text_box.insert(tk.END, f"🎙️ Listening in language: {language}\n")
        text_box.see(tk.END)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            text_box.insert(tk.END, "⏱️ Timeout. Please speak again.\n")
            return

    try:
        command = recognizer.recognize_google(audio, language=language).lower()
        text_box.insert(tk.END, f"Command: {command}\n")
        text_box.see(tk.END)
    except sr.UnknownValueError:
        text_box.insert(tk.END, "Could not understand. Try again.\n")
        return
    except sr.RequestError as e:
        text_box.insert(tk.END, f"API Error: {e}\n")
        return

    # === Command Handling ===
    if "switch to english" in command or "ਅੰਗਰੇਜ਼ੀ" in command or "अंग्रेजी" in command:
        language = "en-IN"
        lang_var.set(language)
        speak_in_language("Switched to English")

    elif "switch to hindi" in command or "ਹਿੰਦੀ" in command or "हिंदी" in command:
        language = "hi-IN"
        lang_var.set(language)
        speak_in_language("हिंदी में बदल दिया गया")

    elif "switch to punjabi" in command or "ਪੰਜਾਬੀ" in command or "पंजाबी" in command:
        language = "pa-IN"
        lang_var.set(language)
        speak_in_language("ਪੰਜਾਬੀ ਵਿੱਚ ਬਦਲ ਦਿੱਤਾ ਗਿਆ")

    elif "sat sri akal" in command:
        speak_in_language("Sat Sri Akaal")

    elif "start drawing" in command or "ਚਿੱਤਰ ਬਣਾਓ" in command or "चित्र बनाओ" in command:
        speak_in_language("Starting drawing mode")
        threading.Thread(target=hand_gesture_drawing).start()

    elif "stop drawing" in command or "ਚਿੱਤਰ ਰੋਕੋ" in command or "चित्र बंद करो" in command:
        drawing_enabled = False
        speak_in_language("Drawing stopped")

def hand_gesture_drawing():
    global drawing_enabled
    drawing_enabled = True

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)

    canvas = None
    prev_x, prev_y = None, None

    while drawing_enabled:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                cx, cy = int(index_finger.x * w), int(index_finger.y * h)

                if prev_x is not None and prev_y is not None:
                    cv2.line(frame, (prev_x, prev_y), (cx, cy), (255, 0, 255), 4)
                prev_x, prev_y = cx, cy

        else:
            prev_x, prev_y = None, None

        cv2.imshow("Hand Drawing Mode", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    drawing_enabled = False

listen_button = tk.Button(root, text="🎤 Speak", command=lambda: threading.Thread(target=listen_for_commands).start())
listen_button.pack(pady=10)

root.mainloop()
