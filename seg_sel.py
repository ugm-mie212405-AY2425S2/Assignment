import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image and check if it's loaded properly
image = cv2.imread("E:/VSProject/imagesel/sel5.jpg")
if image is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply median and Gaussian filters for noise reduction
blurM = cv2.medianBlur(gray, 5)
blurG = cv2.GaussianBlur(gray, (9, 9), 2)

# Apply histogram equalization
histoNorm = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
claheNorm = clahe.apply(gray)

# Contrast stretching
def pixelVal(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return (s1 / r1) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

r1, s1, r2, s2 = 70, 0, 200, 255
pixelVal_vec = np.vectorize(pixelVal)
contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2)

# Edge detection using Canny
edge = cv2.Canny(gray, 100, 200)
edgeG = cv2.Canny(blurG, 100, 200)
edgeM = cv2.Canny(blurM, 100, 200)

# Morphological operations
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=1)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# Adaptive Thresholding
th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
ret4, th4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Circle detection using Hough Transform
circles = cv2.HoughCircles(blurG, cv2.HOUGH_GRADIENT, 1.2, 20,
                           param1=50, param2=28, minRadius=1, maxRadius=20)

display = image.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(display, (x, y), r, (0, 255, 0), 2)
        cv2.circle(display, (x, y), 2, (0, 0, 255), 3)

# Plot results using Matplotlib
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.ravel()

titles = ['Original', 'Grayscale', 'Median Blur', 'Gaussian Blur',
          'Histogram Equalization', 'CLAHE', 'Contrast Stretch', 'Canny Edge',
          'Dilation', 'Closing', 'Adaptive Mean', 'Adaptive Gaussian']

images = [image, gray, blurM, blurG,
          histoNorm, claheNorm, contrast_stretched, edge,
          dilation, closing, th2, th3]

for i in range(12):
    if len(images[i].shape) == 2:
        axes[i].imshow(images[i], cmap='gray')
    else:
        axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Show detected circles separately
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
plt.title("Detected Circles")
plt.axis('off')
plt.show()
