import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class WatershedClassifier():
    def __init__(self, filepath):
        self.filepath = filepath

    def forward(self):
        img = cv.imread(self.filepath)
        self.img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Otsu's thresholding on grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, self.otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Noise removal with morphological opening
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        opened = cv.morphologyEx(self.otsu, cv.MORPH_OPEN, kernel, iterations=2)

        # Sure background and foreground
        sure_bg = cv.dilate(opened, kernel, iterations=3)
        dist = cv.distanceTransform(opened, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Unknown region and markers
        unknown = cv.subtract(sure_bg, sure_fg)
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed
        self.result = self.img_rgb.copy()
        markers = cv.watershed(img, markers)
        self.result[markers == -1] = [255, 0, 0]  # boundary in red

        return markers

    def plot(self):
        self.forward()
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        axes[0].imshow(self.img_rgb);      axes[0].set_title('Original')
        axes[1].imshow(self.otsu, cmap='gray'); axes[1].set_title("Otsu's Threshold")
        axes[2].imshow(self.result);       axes[2].set_title('Watershed')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()