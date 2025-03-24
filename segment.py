import cv2 as cv
import numpy as np


def segment_blood(image_path):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 70, 150)

    kernel = np.ones((2, 2), np.uint8)
    dilated = cv.dilate(edges, kernel, iterations=1)

    erd = cv.erode(dilated, kernel, iterations=1)

    contour, _ = cv.findContours(erd, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv.drawContours(mask, contour, -1, (255), thickness=cv.FILLED)

    kernel2 = np.ones((7, 7), np.uint8)
    eroded2 = cv.erode(mask, kernel2, iterations=1)

    kernel3 = np.ones((6, 6), np.uint8)
    dlt = cv.dilate(eroded2, kernel3, iterations=1)

    final = np.zeros_like(dlt)
    contour, _ = cv.findContours(dlt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contour:
        area = cv.contourArea(contour)
        if area >= 150:
            cv.drawContours(final, [contour], -1, (255), thickness=cv.FILLED)
            count += 1
    print(f"Number of blood cells detected: {count}")

    segmented = cv.bitwise_and(image, image, mask=final)
    return image, eroded2, segmented


image_path = 'data/image.jpg'
image, dilated2, segmented = segment_blood(image_path)

cv.imshow('Original Image', image)
cv.imshow('Dilated Image', dilated2)
cv.imshow('Segmented Image', segmented)
cv.waitKey(0)
cv.destroyAllWindows()
