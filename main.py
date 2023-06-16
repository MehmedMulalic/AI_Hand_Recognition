import Layers
import cv2
import numpy as np

model = Layers.Model.load('hand_gesture_model.model')
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

cap = cv2.VideoCapture(0)
x1, y1, x2, y2 = 300, 50, 610, 360
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    lower_skin = np.array([75, 0, 47], dtype=np.uint8)
    upper_skin = np.array([255, 144, 141], dtype=np.uint8)
    
    #define region of interest
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 0)
    roi = frame[y1:y2, x1:x2]
    
    # Filters and Image Processing
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, np.ones((1, 1), np.uint8), iterations=4)
    mask = cv2.GaussianBlur(mask, (1, 1), 1)

    _, thresholded = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresholded, (50, 50))
    resized_array = resized.reshape(-1, 2500)

    # ANN
    confidences = model.predict(resized_array)
    predictions = model.output_layer_activation.predictions(confidences)
    result = labels[predictions[0]]

    # Display
    cv2.putText(frame, str(result), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (54, 69, 79), 3)
    cv2.imshow('Frame', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


# REFERENCE
# https://github.com/Sadaival/Hand-Gestures/blob/master/gesture.py