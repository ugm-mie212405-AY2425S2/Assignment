import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('seldarah.png')

green_channel = image[:, :, 2]

green_blur = cv2.GaussianBlur(green_channel, (5, 5), 0)

edges = cv2.Canny(green_blur, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_mask = np.zeros_like(green_channel)
cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)

segmented_result = image.copy()
cv2.drawContours(segmented_result, contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(232), plt.imshow(green_channel, cmap='gray'), plt.title('Green Channel')
plt.subplot(233), plt.imshow(edges, cmap='gray'), plt.title('Canny Edge (Red)')
plt.subplot(234), plt.imshow(closing, cmap='gray'), plt.title('Morphology Closing')
plt.subplot(235), plt.imshow(filled_mask, cmap='gray'), plt.title('Filled Mask')
plt.subplot(236), plt.imshow(cv2.cvtColor(segmented_result, cv2.COLOR_BGR2RGB)), plt.title('Segmented Contours')

plt.tight_layout()
plt.show()

cv2.imwrite('segmented_cells.png', segmented_result)
