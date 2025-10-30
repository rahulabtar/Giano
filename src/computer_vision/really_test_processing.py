import enum
import cv2 as cv
import numpy as np

test_1 = cv.imread("src/computer_vision/calibration_output/calibration_20251016_200353/03_grayscale.jpg", cv.IMREAD_GRAYSCALE)
test_2 = cv.imread("src/computer_vision/calibration_output/calibration_20251016_200928/03_grayscale.jpg", cv.IMREAD_GRAYSCALE)
test_3 = cv.imread("src/computer_vision/calibration_output/calibration_20251016_201056/03_grayscale.jpg", cv.IMREAD_GRAYSCALE)

test_images = [test_1, test_2, test_3]
c_values = [i/2 for i in range(2, 6)]
all_thresholded = []
for image in test_images:
  thresholded_images = []
  for c in c_values:
    block_size = 11
    med_image = cv.medianBlur(image, 5)
    test_image = cv.adaptiveThreshold(med_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, c)
    thresholded_images.append(test_image)
    cv.imshow(f"Threshold with block size {block_size} and C value {c}", test_image)
  all_thresholded.append(thresholded_images)


while True:
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cv.destroyAllWindows()

height, width = test_1.shape
for image_list in all_thresholded:
  for i, image in enumerate(image_list):
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (width//4, 1))
    horizontal_lines = cv.morphologyEx(image, cv.MORPH_OPEN, horizontal_kernel)
      
      # Detect vertical lines for key boundaries
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, height//8))
    vertical_lines = cv.morphologyEx(image, cv.MORPH_OPEN, vertical_kernel)

    # Combine horizontal and vertical lines
    combined_lines = cv.add(horizontal_lines, vertical_lines)

    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,5))
    open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

    closed = cv.morphologyEx(image, cv.MORPH_CLOSE, close_kernel)
    open = cv.morphologyEx(closed, cv.MORPH_OPEN, open_kernel)
    cv.imshow(f"balls in my face{i}", open)

while True:
  if cv.waitKey(1) & 0xFF == ord('q'):
    cv.destroyAllWindows()
    break

contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


"""from skimage.filters import threshold_multiotsu

# Calculate three thresholds (background, black keys, white keys)
thresholds = threshold_multiotsu(test_1, classes=3)

# Create regions based on thresholds
regions = np.digitize(test_1, bins=thresholds)

# Extract black keys (middle intensity class)
black_keys_mask = (regions == 1).astype(np.uint8) * 255

# Extract white keys (highest intensity class)  
white_keys_mask = (regions == 2).astype(np.uint8) * 255

# Combine masks
combined_mask = cv.bitwise_or(white_keys_mask, black_keys_mask)

cv.imshow("combined mask multiotsu", combined_mask)
while True:
  if cv.waitKey(1) & 0xFF == ord('q'):
    cv.destroyAllWindows()
    break"""