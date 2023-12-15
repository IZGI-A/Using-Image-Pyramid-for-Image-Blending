import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

class Blend:
    def __init__(self):
        self.image_size = 512
        pass

    def resize_image(self, image):
        return cv2.resize(image, (self.image_size, self.image_size))

    def generate_gaussian_pyramid(self, image, levels):
        pyramid = [image.copy()]
        for i in range(levels):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def generate_laplacian_pyramid(self, gaussian_pyramid):
        laplacian_pyramid = [gaussian_pyramid[-1]]
        for i in range(len(gaussian_pyramid) - 1, 0, -1):
            gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
            laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
            laplacian_pyramid.append(laplacian)
        return laplacian_pyramid

    def blended_images(self, lp_a, lp_b, masks):
        blended_pyramid = []
        for la, lb, mask in zip(lp_a, lp_b, masks):
            mask = cv2.resize(mask, (la.shape[1], la.shape[0]))
            mask_normalized = mask / 255.0
            blended = la * mask_normalized + lb * (1 - mask_normalized)
            blended_pyramid.append(blended)
        return blended_pyramid

    def reconstruct_image(self, pyramid):
        reconstructed_image = pyramid[0].astype(np.float32)
        reconstructed_image_array = []
        for i in range(1, len(pyramid)):
            reconstructed_image = cv2.pyrUp(reconstructed_image)
            pyramid[i] = pyramid[i].astype(np.uint8)
            reconstructed_image = cv2.add(pyramid[i], reconstructed_image, dtype=cv2.CV_8U)
            reconstructed_image = cv2.normalize(reconstructed_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            reconstructed_image_array.append(reconstructed_image.astype(np.uint8))
        cv2.imshow(f"Final Image {i}", reconstructed_image.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return reconstructed_image_array

    def display_pyramid_levels(self, images, title):
        num_levels = len(images)
        plt.figure(figsize=(10, 8))

        for i in range(num_levels):
            plt.subplot(2, num_levels, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title(f'{title} L- {i}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Pyramid Blender')
    parser.add_argument('image1_path', type=str, help='Path to first image')
    parser.add_argument('image2_path', type=str, help='Path to second image')
    parser.add_argument('num_levels', type=int, help='Number of pyramid levels')
    args = parser.parse_args()

    blender = Blend()

    image1 = cv2.imread(args.image1_path)
    image2 = cv2.imread(args.image2_path)

    image1 = blender.resize_image(image1)
    image2 = blender.resize_image(image2)

    num_levels = args.num_levels
    max_level = math.log2(blender.image_size)
    if num_levels > max_level:
        print(f"write a value smaller or equal to {max_level} ")
    im1_gaus = blender.generate_gaussian_pyramid(image1, num_levels)
    im2_gaus = blender.generate_gaussian_pyramid(image2, num_levels)

    laplacian1 = blender.generate_laplacian_pyramid(im1_gaus)
    laplacian2 = blender.generate_laplacian_pyramid(im2_gaus)

    mask = np.zeros_like(image1)
    rect_roi = cv2.selectROI('Select ROI', image1)
    cv2.destroyWindow('Select ROI')
    x, y, w, h = rect_roi
    mask[y:y + h, x:x + w] = 255
    mask_pyramid_ = blender.generate_gaussian_pyramid(mask, num_levels)
    mask_pyramid = mask_pyramid_[::-1]

    # Display levels using subplot
    blender.display_pyramid_levels(im1_gaus, 'Img 1 Gauss')
    blender.display_pyramid_levels(im2_gaus, 'Img 2 Gauss')
    blender.display_pyramid_levels(mask_pyramid_, 'Mask Gauss')
    blender.display_pyramid_levels(laplacian1, 'Img 1 Lap')
    blender.display_pyramid_levels(laplacian2, 'Img 2 Lap')

    blended_pyramid = blender.blended_images(laplacian1, laplacian2, mask_pyramid)
    blender.display_pyramid_levels(blended_pyramid, 'Blended')
    result = blender.reconstruct_image(blended_pyramid)
    blender.display_pyramid_levels(result, 'Result')

