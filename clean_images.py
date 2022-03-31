import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import array, float32
from numpy import asarray
from skimage.filters import try_all_threshold
from skimage.filters import threshold_local
from skimage.feature import canny
from skimage.exposure import equalize_hist
from skimage.exposure import equalize_adapthist
from skimage.segmentation import slic
from skimage.measure import find_contours
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import corner_harris
from skimage.feature import corner_peaks
from PIL import ImageFilter


class ImageClense:
    """
    This class provides key functionality to extract and manipulate image data.
    It has been created with the primarily use case of image classification.

    Args:
        download (str): The directory the images are stored
        upload (str): The directory the images will be uploaded too once prepared
    """
    def __init__(self, download, upload) -> None:
        self.download = os.listdir(download)
        self.upload = upload
        """See help(ImageCleanse) for accurate signature."""
        pass

    @staticmethod
    def resize_image_to_square(image: Image, length: int) -> Image:
        """
        Resize an image to a square. Can make an image bigger to make it fit
        or smaller if it doesn't fit. It also crops part of the image.

        credit to Titouan
        https://stackoverflow.com/questions/43512615/reshaping-rectangular-image-to-square

        Args:
            image (Image): Image to resize.
            length (int): Width and height of the output image.

        Returns
            resized_image (Image): The image with new dimensions
        """

        """
        Resizing strategy : 
        1) We resize the smallest side to the desired dimension (e.g. 1080)
        2) We crop the other side so as to make it fit with the same length as the smallest side (e.g. 1080)
        """
        if image.size[0] < image.size[1]:
            # The image is in portrait mode. Height is bigger than width.
            # This makes the width fit the LENGTH in pixels while conserving the ration.
            resized_image = image.resize((length, int(image.size[1] * (length / image.size[0]))))
            # Amount of pixel to lose in total on the height of the image.
            required_loss = (resized_image.size[1] - length)
            # Crop the height of the image so as to keep the center part.
            resized_image = resized_image.crop(
                box=(0, required_loss / 2, length, resized_image.size[1] - required_loss / 2))
            # We now have a length*length pixels image.
            return resized_image
        else:
            # This image is in landscape mode or already squared. The width is bigger than the heihgt.
            # This makes the height fit the LENGTH in pixels while conserving the ration.
            resized_image = image.resize((int(image.size[0] * (length / image.size[1])), length))
            # Amount of pixel to lose in total on the width of the image.
            required_loss = resized_image.size[0] - length
            # Crop the width of the image so as to keep 1080 pixels of the center part.
            resized_image = resized_image.crop(
                box=(required_loss / 2, 0, resized_image.size[0] - required_loss / 2, length))
            # We now have a length*length pixels image.
            return resized_image

    def get_images(self, directory):
        """
        """
        img_list = []
        file_names = []
        for file in self.download:
            img = Image.open(os.path.join(directory, file))
            img = self.resize_image_to_square(img, 256)
            img_list.append(img)
            file_names.append(file)
        return img_list, file_names
    
    @staticmethod
    def image_to_array(img):
        """
        """
        img = np.asarray(img)
        return img
    
    @staticmethod
    def image_to_greyscale(img):
        """
        """
        greyscale = img.convert(mode='L')
        return greyscale
    
    def normalise_images(self, img):
        img = self.image_to_array(img)
        pixels = img.astype('float32')
        pixels /= 255
        return pixels

    @staticmethod
    def show_image(image: np.ndarray, title="Image", cmap_type="gray", axis=False):
        """
        A function to display np.ndarrays as images
        """
        plt.imshow(image, cmap=cmap_type)
        plt.title(title)
        if not axis:
            plt.axis("off")
        plt.margins(0, 0)
        plt.show()
    
    def mark_contours(image):
        """A function to find contours from an image"""
        gray_image = rgb2gray(image)
        # Find optimal threshold
        thresh = threshold_otsu(gray_image)
        # Mask
        binary_image = gray_image > thresh

        contours = find_contours(binary_image)

        return contours
    
    def plot_image_contours(image):
        fig, ax = plt.subplots()

        ax.imshow(image, cmap=plt.cm.gray)

        for contour in mark_contours(image):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="red")

        ax.axis("off")
    
    def manual_global_threshold(self, img, reversed=True):
        """
        Works best with greyscale images.
        """
        normalised = self.normalise_images(img)
        threshold = normalised.mean()
        if reversed == True:
            binary_array = normalised > threshold
        else:
            binary_array = normalised <= threshold
        binary_image = Image.fromarray(binary_array)
        return binary_image
    
    def try_all_global_threshold(self, img):
        """
        See 
        """
        img_array = self.image_to_array(img)
        fig, ax = try_all_threshold(img_array, figsize=(10, 8), verbose=False)
        return fig, ax
    
    @staticmethod
    def auto_global_threshold(img, threshold):
        """
        See 
        """
        threshold_img = threshold(img)
        return threshold_img

    def manaual_local_threshold(self, img, block_size: int = 5, offset: float32 = 2.000, reversed=True):
        img_array = self.image_to_array(img)
        local_thresh = threshold_local(img_array, block_size= block_size, offset= offset)
        if reversed == True:
            local_binary = img_array > local_thresh
        else:
            local_binary = img_array <= local_thresh
        local_binary_img = Image.fromarray(local_binary)
        return local_binary_img

    def edge_detection(self, img, sigma=1):
        img_array = self.image_to_array(img)
        img_edge = canny(img_array, sigma=sigma)
        img_edge = Image.fromarray(img_edge)
        return img_edge
    
    def denoise_image(self, img, denoise_type=denoise_tv_chambolle, weight=0.3):
        img_array = self.image_to_array(img)
        denoised_img = denoise_type(img_array, channel_axis=True, weight=weight)
        denoised_img = Image.fromarray(np.uint8(denoised_img)).convert('RGB')
        return denoised_img

    def corner_detection(self, img, pixel_distance=50, denoise=True):
        img_array = self.image_to_array(img)
        measured_img = corner_harris(img_array)
        if denoise == True:
            measured_img = denoise_tv_chambolle(measured_img, channel_axis=True, weight=0.3)
        corner_coords = corner_peaks(measured_img, min_distance=pixel_distance)
        return corner_coords
    
    def plot_corner_detection(self, corner_coords, img):
        img_array = self.image_to_array(img)
        plt.imshow(img_array, cmap="gray")
        plt.plot(corner_coords[:, 1], corner_coords[:, 0], "+b", markersize=15)
        plt.axis("off")
    
    def centreing(self, pixels, global_centre=True, noramlised=True):
        # Consider a function that allows user to pick whther to centre pre or post normalisation with normalised yes/no
        # Centre prior to normalisation 
        # global_centered = pixels
        # global_centered = global_centered - pixel_mean
        # global_centered = global_centered.mean()
        pass
    
    @staticmethod
    def save_images(path, folder, img, file_name):
        full_path = os.path.join(path, folder)
        if os.path.exists(full_path) != True:
            os.makedirs(full_path)
        img.save(os.path.join(full_path, file_name), format='JPEG')
    
    @staticmethod
    def erode(cycles, img):
        """
        https://realpython.com/image-processing-with-the-python-pillow-library/#image-segmentation-and-superimposition-an-example
        """
        for _ in range(cycles):
            img = img.filter(ImageFilter.MinFilter(3))
        return img

    @staticmethod
    def dilate(cycles, img):
        for _ in range(cycles):
            img = img.filter(ImageFilter.MaxFilter(3))
        return img
    
    def erode_dilate(self, img, erode_cycles, dilate_cycles, reverse=True):
        img_array = self.image_to_array(img)
        pixels = img_array.astype('float32')
        pixel_mean = pixels.mean()
        if reverse == True:
            if pixel_mean > 0.5:
                img = self.erode(erode_cycles, img)
            else:
                img = self.dilate(dilate_cycles, img)
        else:
            if pixel_mean > 0.5:
                img = self.dilate(dilate_cycles, img)
            else:
                img = self.erode(erode_cycles, img)
        return img    

    # Consider functions for perform Data Augementation
    # Rotate, at an angle and Horizontal
    # Change brightness randomly

if __name__ == "__main__":
    download_image_directory = os.path.join(os.getcwd(),'images_fb/test_images/')
    upload_image_directory = os.path.join(os.getcwd(),'images_fb/cleaned_test_images/')
    cleanse = ImageClense(download_image_directory, upload_image_directory)
    resized_images, image_names = cleanse.get_images(download_image_directory)
    img_number = 0
    for img in resized_images:
        file = image_names[img_number]
        original_array = cleanse.image_to_array(img)
        greyscale = cleanse.image_to_greyscale(img)
        binary = cleanse.manual_global_threshold(greyscale)
        binary_reversed = cleanse.manual_global_threshold(greyscale, False)
        local_threshold = cleanse.manaual_local_threshold(greyscale, 5, 2)
        edge_detection = cleanse.edge_detection(greyscale)
        eroded_binary = cleanse.erode(1, binary_reversed)
        blank = img.point(lambda _: 0)
        eroded_extract = Image.composite(greyscale, blank, eroded_binary)
        dilated_binary = cleanse.dilate(1, binary_reversed)
        erode_dilate = cleanse.erode_dilate(binary_reversed, 10, 2, False)
        eroded_dilated_extract = Image.composite(greyscale, blank, erode_dilate)




        # https://coderzcolumn.com/tutorials/python/image-filtering-in-python-using-pillow
        contour = img.filter(ImageFilter.CONTOUR)
        grey_contour = greyscale.filter(ImageFilter.CONTOUR)
        gaussian_blur = img.filter(ImageFilter.GaussianBlur)
        detailed = img.filter(ImageFilter.DETAIL)
        edge_enhance = img.filter(ImageFilter.EDGE_ENHANCE)
        embossed = img.filter(ImageFilter.EMBOSS)
        find_edges = img.filter(ImageFilter.FIND_EDGES)
        find_edges_grey = greyscale.filter(ImageFilter.FIND_EDGES)
        sharpened = img.filter(ImageFilter.SHARPEN)
        smoothed = img.filter(ImageFilter.SMOOTH)

        combined  = cleanse.image_to_greyscale(sharpened)
        combined = combined.filter(ImageFilter.SMOOTH)
        combined_threshold = cleanse.manual_global_threshold(combined, False)
        # combined_edges = combined_threshold.filter(ImageFilter.FIND_EDGES)
        combined_final = cleanse.erode_dilate(combined_threshold, 10, 2, False)
        combined_final = Image.composite(greyscale, blank, combined_final)

        cleanse.save_images(cleanse.upload, 'original', img, file)
        cleanse.save_images(cleanse.upload, 'greyscale', greyscale, file)
        cleanse.save_images(cleanse.upload, 'binary', binary, file)
        cleanse.save_images(cleanse.upload, 'binary_reversed', binary_reversed, file)
        cleanse.save_images(cleanse.upload, 'local_threshold', local_threshold, file)
        cleanse.save_images(cleanse.upload, 'edge_detection', edge_detection, file)
        cleanse.save_images(cleanse.upload, 'contour', contour, file)
        cleanse.save_images(cleanse.upload, 'grey_contour', grey_contour, file)
        cleanse.save_images(cleanse.upload, 'gaussian_blur', gaussian_blur, file)
        cleanse.save_images(cleanse.upload, 'detailed', detailed, file)
        cleanse.save_images(cleanse.upload, 'edge_enhance', edge_enhance, file)
        cleanse.save_images(cleanse.upload, 'embossed', embossed, file)
        cleanse.save_images(cleanse.upload, 'find_edges', find_edges, file)
        cleanse.save_images(cleanse.upload, 'find_edges_grey', find_edges_grey, file)
        cleanse.save_images(cleanse.upload, 'sharpened', sharpened, file)
        cleanse.save_images(cleanse.upload, 'smoothed', smoothed, file)
        cleanse.save_images(cleanse.upload, 'eroded_binary', eroded_binary, file)
        cleanse.save_images(cleanse.upload, 'dilated_binary', dilated_binary, file)
        cleanse.save_images(cleanse.upload, 'eroded_extract', eroded_extract, file)
        cleanse.save_images(cleanse.upload, 'combined', combined_threshold, 'thres_'+file)
        # cleanse.save_images(cleanse.upload, 'combined', combined_edges, 'edge_'+file)
        cleanse.save_images(cleanse.upload, 'combined', combined_final, 'final_'+file)
        img_number += 1
