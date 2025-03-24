import cv2
import numpy as np
from matplotlib import pyplot as plt

def segment_cells_watershed(image_paths):
    plt.figure(figsize=(15, 10))
    
    for i, image_path in enumerate(image_paths):
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        ret, thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Determine sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=1)
        
        # Determine sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        
        # Determine unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]  # Mark boundaries in blue
        
        # Display results
        plt.subplot(3, 2, i * 2 + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Segmented Cells {i+1}')
        plt.axis('off')
        
        plt.subplot(3, 2, i * 2 + 2)
        plt.imshow(markers, cmap='jet')
        plt.title(f'Watershed Markers {i+1}')
        plt.axis('off')
    
    plt.show()

image_paths = ['eritrosit1.png', 'eritrosit2.png', 'eritrosit3.png']  
segment_cells_watershed(image_paths)
