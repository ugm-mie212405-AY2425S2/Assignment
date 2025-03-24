import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./data/image.jpg')
if image is None:
    print("Gagal membaca gambar!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
gray = cv2.equalizeHist(gray)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Image')
plt.subplot(1,2,2), plt.imshow(thresh, cmap='gray'), plt.title("Segmented Image")
plt.show()