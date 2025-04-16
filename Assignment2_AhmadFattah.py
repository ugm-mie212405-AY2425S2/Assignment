#import library
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Pra pemrosesan
image = cv2.imread("rbc.jpg")
green_channel = image[:, :, 1]
filter = cv2.medianBlur(green_channel,9)

#Segmentasi
_, thresh = cv2.threshold(filter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Filling hole menggunakan Flood Fill untuk mengisi lubang yang disebabkan oleh cekungan sel darah merah

#Mendapatkan ukuran citra (height, width)
h, w = thresh.shape
print(h, w)

# Mendefinisikan mask yang akan digunakan untuk proses flood fill dengan height & width +2 dari ukuran aslinya
mask = np.zeros((h+2, w+2), np.uint8)

# Mendefinisikan bentuk yang telah didapat dari thresholding sebagai citra yang akan ditumpuk selama proses flood fill
flood_filled = thresh.copy()

# Parameter ketiga merupakan posisi dimulainya flood fill. 
# Posisi ini perlu diletakkan ke background citra. Apabila terletak pada objek, flood fill akan gagal.
# Parameter keempat untuk menunjukkan warna dari flood fill. Biasanya bernilai 255, tapi dapat diubah untuk menunjukkan warna hasil floodfill yang lebih baik
cv2.floodFill(flood_filled, mask, (150, 0), 255)


# Inversi hasilnya agar mendapatkan bentuk penuh sel darah merah
filled_holes = cv2.bitwise_not(flood_filled)

# Menggabungkan hasil filling hole dengan hasil segmentasi awal
final_segmented = cv2.bitwise_or(thresh, filled_holes)

# Morphological Closing untuk memperbaiki batas objek
kernel = np.ones((3,3), np.uint8)
closed = cv2.morphologyEx(final_segmented, cv2.MORPH_CLOSE, kernel, iterations=2)

plt.title("citra asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

plt.title("hasil filter green channel")
plt.imshow(filter)
plt.show()

plt.title("thresh")
plt.imshow(thresh, cmap="gray")
plt.show()

plt.title("mask")
plt.imshow(mask, cmap="gray")
plt.show()

plt.title("flood")
plt.imshow(flood_filled, cmap="gray")
plt.show()

plt.title("filled_holes")
plt.imshow(filled_holes, cmap="gray")
plt.show()

plt.title("final")
plt.imshow(closed, cmap="gray")
plt.show()