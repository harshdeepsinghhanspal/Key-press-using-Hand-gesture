import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize PyAutoGUI
pyautogui.FAILSAFE = False

# Open the webcam
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
prev_time = 0
fps = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Mirror the image horizontally
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe
        results = hands.process(image_rgb)

        # Draw a line in the middle of the image
        height, width, _ = image.shape
        cv2.line(image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

        # Add Labels
        cv2.putText(image, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "RIGHT", (width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process the image with MediaPipe
        results = hands.process(image_rgb)

        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of index finger
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x = int(index_finger.x * width)

                # Check if index finger is on the left side of the screen
                if index_finger_x < width // 3:
                    pyautogui.keyDown('left')
                    pyautogui.keyUp('right')
                    print("LEFT")

                # Check if index finger is on the right side of the screen
                elif index_finger_x > 2 * width // 3:
                    pyautogui.keyDown('right')
                    pyautogui.keyUp('left')
                    print("RIGHT")

                else:
                    pyautogui.keyUp('left')
                    pyautogui.keyUp('right')

        # Calculate FPS
        curr_time = time.time()
        if curr_time - prev_time > 0:
            fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Display FPS at the bottom left corner
        cv2.putText(image, f"FPS: {fps}", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the image
        cv2.imshow('Hand Tracking', image)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
