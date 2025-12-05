import cv2
import numpy as np

# Open webcam
webcam = cv2.VideoCapture(0)
kernel = np.ones((5, 5), "uint8")

# Define HSV ranges for ROYGB
color_ranges = {
    "Red":    [([0, 100, 50], [10, 255, 255]), ([170, 100, 50], [180, 255, 255])],  # wrap-around
    "Orange": [([11, 100, 50], [25, 255, 255])],
    "Yellow": [([26, 100, 50], [34, 255, 255])],
    "Green":  [([35, 100, 50], [85, 255, 255])],
    "Blue":   [([100, 100, 50], [130, 255, 255])]
}

# BGR colors for drawing rectangles
bgr_colors = {
    "Red": (0, 0, 255),
    "Orange": (0, 165, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0)
}

# Minimum and maximum contour area
MIN_AREA = 300
MAX_AREA = 5000  # adjust as needed

# Optional bounding box size limits
MIN_WIDTH, MIN_HEIGHT = 20, 20
MAX_WIDTH, MAX_HEIGHT = 200, 200

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # optional horizontal flip
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, ranges in color_ranges.items():
        mask = None
        # Handle multiple ranges (for red)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            if mask is None:
                mask = cv2.inRange(hsv, lower_np, upper_np)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

        # Dilate mask
        mask = cv2.dilate(mask, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                if MIN_WIDTH < w < MAX_WIDTH and MIN_HEIGHT < h < MAX_HEIGHT:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_colors[color_name], 2)
                    cv2.putText(frame, f"{color_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_colors[color_name], 2)

    cv2.imshow("5-Color Detection (ROYGB)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
