import cv2
import dlib
import math
import numpy as np
from collections import deque
from constants import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("face_weights.dat")

# Initialize video capture with camera index 0
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Adjustable threshold for lip distance
lip_distance_threshold = 45
# Buffer to store recent talking/not talking states
state_buffer = deque(maxlen=10)
curr_word_frames = []
not_talking_counter = 0
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    print("Frame read successfully")

    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    print("Frame converted to grayscale")

    # Use detector to find landmarks
    faces = detector(gray)
    print(f"Faces detected: {len(faces)}")

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)
        print("Landmarks detected")

        # Calculate the distance between the upper and lower lip landmarks
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])

        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        # Add padding if necessary to get a 76x110 frame
        width_diff = LIP_WIDTH - (lip_right - lip_left)
        height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top

        # Ensure that the padding doesn't extend beyond the original frame
        pad_left = min(pad_left, lip_left)
        pad_right = min(pad_right, frame.shape[1] - lip_right)
        pad_top = min(pad_top, lip_top)
        pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

        # Create padded lip region
        lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

        lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
        # Apply contrast stretching to the L channel of the LAB image
        l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
        l_channel_eq = clahe.apply(l_channel)

        # Merge the equalized L channel with the original A and B channels
        lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
        lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
        lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

        # Apply the kernel to the input image
        lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
        lip_frame = lip_frame_eq

        # Draw a circle around the mouth
        for n in range(48, 61):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # Determine if the person is talking based on the lip distance
        is_talking = lip_distance > lip_distance_threshold
        state_buffer.append(is_talking)

        # Only change the state if the new state is consistent over the buffer
        if state_buffer.count(True) > len(state_buffer) // 2:
            current_state = "Talking"
            color = (0, 255, 0)
        else:
            current_state = "Not talking"
            color = (0, 0, 255)

        # Display the current state
        cv2.putText(frame, current_state, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    # Display the captured frame
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # Exit when escape is pressed
    if key == 27:
        break

cap.release()

# Close all windows
cv2.destroyAllWindows()
