import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def img_preproc(path):
    """
    Load and preprocess the image: convert to grayscale
    and apply Gaussian Blur.
    :param path: Path to the image file.
    :return: Tuple of original, grayscale, and blurred images.
    """
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError("Image file not found. Please check the path.")

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 3)
    return img, img_gray, img_blur


def segment_cells(img_gray):
    """
    Apply segmentation using thresholding and morphological operations.
    :param img_gray: Grayscale image.
    :return: Processed images for visualization and segmentation markers.
    """
    ret, binary = cv.threshold(
        img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    img_close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
    img_dilate = cv.morphologyEx(img_close, cv.MORPH_DILATE, kernel)
    dist_transform = cv.distanceTransform(img_dilate, cv.DIST_L2, 5)
    ret, dist_threshold = cv.threshold(
        dist_transform, 0.3 * dist_transform.max(), 255, 0
    )

    dist_threshold = np.uint8(dist_threshold)
    unknown = cv.subtract(img_dilate, dist_threshold)
    ret, markers = cv.connectedComponents(dist_threshold)
    markers = markers + 1
    markers[unknown == 255] = 0

    return (
        binary, img_close, dist_transform, markers, dist_threshold, img_dilate
    )


def apply_watershed(image, markers):
    """
    Apply watershed algorithm to segment objects.
    :param image: Original image.
    :param markers: Marker image for segmentation.
    :return: Image with watershed segmentation applied.
    """
    markers = np.int32(markers)
    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Color boundaries in blue
    return image


if __name__ == "__main__":
    """
    Data Acquisition Explanation
    Data Acquisition: The dataset should contain cell images captured under
    a microscope.
    Images should be in standard formats (PNG/JPEG) and stored in a directory.
    """

    img_path = "sample_image.png"  # Update with your actual file path

    try:
        img, img_gray, img_blur = img_preproc(img_path)
    except FileNotFoundError as e:
        print(e)
        exit()

    result = segment_cells(img_blur)
    binary, closing, dist_transform, markers, sure_fg, sure_bg = result
    result_image = apply_watershed(img.copy(), markers)

    # Display results
    plt.figure()
    plt.suptitle("Segmentation Process")
    plt.subplot(231)
    plt.title("Binary Image")
    plt.imshow(binary, cmap='gray')
    plt.subplot(232)
    plt.title("Closing Image")
    plt.imshow(closing, cmap='gray')
    plt.subplot(233)
    plt.title("Dilate Image")
    plt.imshow(sure_bg, cmap='gray')
    plt.subplot(234)
    plt.title("Distance Transform")
    plt.imshow(dist_transform, cmap='gray')
    plt.subplot(235)
    plt.title("Distance Threshold")
    plt.imshow(sure_fg, cmap='gray')
    plt.subplot(236)
    plt.title("Markers")
    plt.imshow(markers, cmap='jet')

    plt.figure()
    plt.suptitle("Final Result")
    plt.imshow(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
    plt.show()
