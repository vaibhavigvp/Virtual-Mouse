import cv2
import mediapipe as mp
import pyautogui

# Initialize the video camera, face mesh model, and screen dimensions
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Define the iris threshold values for right and left click
right_iris_threshold = 0.99
left_iris_threshold = 0.99

# Initialize the last detected iris values
last_right_iris = 0.99
last_left_iris = 0.99

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the face landmarks and iris points
    output = face_mesh.process(rgb_frame)

    if output is None:
        continue

    landmark_points = output.multi_face_landmarks
    iris_points = output.multi_face_landmarks

    # Get the frame dimensions
    frame_h, frame_w, _ = frame.shape

    # Draw circles around the iris points and detect right and left iris positions
    if iris_points:
        irises = iris_points[0].landmark[468:470]
        for id, iris in enumerate(irises):
            x = int(iris.x * frame_w)
            y = int(iris.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 0:
                last_left_iris = iris.x
            elif id == 1:
                last_right_iris = iris.x

        # If both right and left iris are detected, control the mouse
        if last_right_iris and last_left_iris:
            # Move the mouse based on the right iris position
            if last_right_iris > right_iris_threshold:
                screen_x = screen_w - ((screen_w * last_right_iris) * 3)
                screen_y = screen_h * irises[1].y
                pyautogui.moveTo(screen_x, screen_y)
            # Left-click using the left iris position
            elif last_left_iris < left_iris_threshold:
                screen_x = screen_w - ((screen_w * last_left_iris) * 3)
                screen_y = screen_h * irises[0].y
                pyautogui.moveTo(screen_x, screen_y)
                pyautogui.click(button='left')
            # Right-click using the right iris position
            elif last_right_iris < right_iris_threshold:
                screen_x = screen_w - ((screen_w * last_right_iris) * 3)
                screen_y = screen_h * irises[1].y
                pyautogui.moveTo(screen_x, screen_y)
                pyautogui.click(button='right')

    cv2.imshow('iris', frame)
    key = cv2.waitKey(1)
    # Press 'q' to exit the program
    if key == ord('q'):
        break

# Release the video camera and destroy all windows
cam.release()
cv2.destroyAllWindows()
