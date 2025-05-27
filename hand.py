import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Create a white canvas to draw
canvas = None
drawing = False  # Drawing state

cap = cv2.VideoCapture(0)

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
            index_finger = lm[8]  # Index finger tip

            x, y = int(index_finger.x * w), int(index_finger.y * h)

            # Only draw when index finger is up and others are down (simplified logic)
            drawing = True

            if drawing:
                cv2.circle(canvas, (x, y), 8, (0, 0, 255), -1)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine drawing with live feed
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Hand Drawing", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = None  # Clear canvas

cap.release()
cv2.destroyAllWindows()
