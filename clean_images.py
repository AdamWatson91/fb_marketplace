import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from PIL.Image import Image as img
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
from tqdm import tqdm
from typing import Union


class ImageCleanse:
    """
    This class provides key functionality to extract and manipulate image data.
    It has been created with the primarily use case of image classification.

    Args:
        download (str): The directory the images are stored
        upload (str): The directory the images will be uploaded too once 
            prepared
    """
    def __init__(self, download: str, upload: str) -> None:
        self.download = os.listdir(download)
        self.upload = upload
        """See help(ImageCleanse) for accurate signature."""
        pass

    @staticmethod
    def crop_image_to_square(image: img, length: int) -> Image:
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

    @staticmethod
    def resize_image_and_pad(img: img, length: int, pad_mode: str = 'RGB') -> img:
        """
        This resizes an image to a sqaure using the same ratio as original
        image. The remaining pixels are then padded with black pixels.

        Args:
            img (PIL.Image): The image to be transformed.
            length (int): The desired length and width of final image.
            pad_mode (str): Default to RGB so that all images are converted
                                to this mode.
        
        Returns:
            padding (PIL.Image) : The transformed image.
        """
        if pad_mode != 'RGB':
            pad_mode = 'RGB'
        padding = Image.new(mode=pad_mode, size=(length, length))
        img_size = img.size
        max_dim = max(img.size)
        ratio = length / max_dim
        new_img_size = (int(img_size[0] * ratio), int(img_size[1] * ratio))
        resized_img = img.resize(new_img_size)
        padding.paste(resized_img, ((length - new_img_size[0]) // 2, (length - new_img_size[1]) // 2))
        return padding

    def get_images(self, directory: str, img_size: int, pad_mode: str, crop: bool = False):
        """
        This downloads images from a desired directory and transforms
        according to user selection. the options are size and whther
        to pad or crop the image.

        Args:
            directory (str): File directory where images are stored.
            img_size (int): desired img size for square.
            pad_mode (str): desired mode for padding
            crop (bool): Whether to crop (True) the image or pad (False).
                            False is default value.

        """
        img_list = []
        file_names = []
        for file in tqdm(self.download, desc='Opening image files...'):
            img = Image.open(os.path.join(directory, file))
            if crop == True:
                img = self.crop_image_to_square(img, img_size)
            else:
                img = self.resize_image_and_pad(img, img_size, pad_mode)
            img_list.append(img)
            file_names.append(file)
        return img_list, file_names

    @staticmethod
    def image_to_array(img: img) -> np.ndarray:
        """
        Converts image file to numpy array.

        Args:
            img (PIL.Iamge): Image to be transformed

        Returns
            img (np.ndarray): Transformed image as an array
        """
        img = np.asarray(img)
        return img

    @staticmethod
    def image_to_greyscale(img: img) -> img:
        """
        Converts image to greyscale image mode.

        Args:
            img (PIL.Iamge): Image to be transformed

        Returns
            greyscale (Pil.Image): Transformed image in greyscale
        """
        greyscale = img.convert(mode='L')
        return greyscale

    def normalise_images(self, img: img)-> np.ndarray:
        """
        Converts image file to numpy array with normalised pixels.

        Args:
            img (PIL.Iamge): Image to be transformed

        Returns
            pixels (np.ndarray): Transformed image as an array of normalised
                                    pixels.
        """
        img = self.image_to_array(img)
        pixels = img.astype('float32')
        pixels /= 255
        return pixels

    @staticmethod
    def show_image(image: np.ndarray, title: str = "Image", cmap_type: str = "gray", axis: bool =False) -> None:
        """
        A function to display np.ndarrays as images
        """
        plt.imshow(image, cmap=cmap_type)
        plt.title(title)
        if not axis:
            plt.axis("off")
        plt.margins(0, 0)
        plt.show()
    
    def mark_contours(image: img) -> img:
        """A function to find contours from an image"""
        gray_image = rgb2gray(image)
        # Find optimal threshold
        thresh = threshold_otsu(gray_image)
        # Mask
        binary_image = gray_image > thresh
        contours = find_contours(binary_image)
        return contours
    
    def plot_image_contours(image: img) -> None:
        """
        Shows image contours so user can view.
        """
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
        for contour in mark_contours(image):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color="red")
        ax.axis("off")
    
    def manual_global_threshold(self, img: img, reversed: bool = True) -> img:
        """
        Applies global thresholding to the image.
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
    

    def manaual_local_threshold(self, img: int, block_size: int = 5, offset: float32 = 2.000, reversed: bool = True)-> img:
        """
        Applies local thresholding to the image.
        Works best with greyscale images.
        """
        img_array = self.image_to_array(img)
        local_thresh = threshold_local(img_array, block_size= block_size, offset= offset)
        if reversed == True:
            local_binary = img_array > local_thresh
        else:
            local_binary = img_array <= local_thresh
        local_binary_img = Image.fromarray(local_binary)
        return local_binary_img

    def edge_detection(self, img: img, sigma: int = 1) -> img:
        """
        Detect edges in the image and outputs new image.
        """
        img_array = self.image_to_array(img)
        img_edge = canny(img_array, sigma=sigma)
        img_edge = Image.fromarray(img_edge)
        return img_edge
    
    def denoise_image(self, img: img, denoise_type=denoise_tv_chambolle, weight: float = 0.3) -> img:
        """
        Applies denoise to the image and outputs new image.
        """
        img_array = self.image_to_array(img)
        denoised_img = denoise_type(img_array, channel_axis=True, weight=weight)
        denoised_img = Image.fromarray(np.uint8(denoised_img)).convert('RGB')
        return denoised_img

    def corner_detection(self, img: img, pixel_distance: int = 50, denoise: bool = True) -> img:
        """
        Applies corner detection to the image and outputs new image.
        """
        img_array = self.image_to_array(img)
        measured_img = corner_harris(img_array)
        if denoise == True:
            measured_img = denoise_tv_chambolle(measured_img, channel_axis=True, weight=0.3)
        corner_coords = corner_peaks(measured_img, min_distance=pixel_distance)
        return corner_coords
    
    def plot_corner_detection(self, corner_coords, img: img) -> None:
        img_array = self.image_to_array(img)
        plt.imshow(img_array, cmap="gray")
        plt.plot(corner_coords[:, 1], corner_coords[:, 0], "+b", markersize=15)
        plt.axis("off")
    
    @staticmethod
    def save_images(path: str, folder: str, img: img, file_name: str) -> None:
        """
        This saves and image in the desired directory with the desired name.
        it checks if the directory exisits and if it does not it creates it.
        The images will be in JPEG format.

        Args:
            path (str): This is the path used to save images to.
            folder (str): The folder name that the images will be
                            saved to in the path.
            img (PIL.Image): Image to be saved
            file_name (str): The name the image will be saved with.
        """
        full_path = os.path.join(path, folder)
        if os.path.exists(full_path) != True:
            os.makedirs(full_path)
        if img.mode == 'RGBA':
            img.convert('RGB')
        try:
            img.save(os.path.join(full_path, file_name), format='JPEG')
        except OSError:
            print(f'Error saving file of type {img.mode}')
            
    @staticmethod
    def erode(cycles: int, img: img) -> img:
        """
        https://realpython.com/image-processing-with-the-python-pillow-library/#image-segmentation-and-superimposition-an-example
        """
        for _ in range(cycles):
            img = img.filter(ImageFilter.MinFilter(3))
        return img

    @staticmethod
    def dilate(cycles: int, img: img) -> img:
        for _ in range(cycles):
            img = img.filter(ImageFilter.MaxFilter(3))
        return img
    
    def erode_dilate(self, img: img, erode_cycles: int, dilate_cycles: int, reverse: bool = True) -> img:
        """
        This function applies erosion and dilation to the image.

        Args:
            img (PIL.Iamge): Image to be transformed
            erode_cycles (int): The number of times to apply erosion
            dilate_cycles (int): The number of times to apply dilation
            reverse (bool): Whether to apply erosion or dilation first.
                                True would erode first
        """
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

    @staticmethod
    def create_holdout(img_list: list, hold_out_size: float) -> img:
        """
        This function randomly splits the list of images into two sets. 

        Args:
            img_list (list): list of PIL Images that are holdout will be
                                created from.
            hold_out_size (float): the percentage size of the total img_list
                                    length that you would like to use as a
                                    holdout.
        Returns:
            holdout (list): list of PIL images for the holdout
            data (list): list of PIL images with the holdout images removed
        """
        list_size = len(img_list)
        random.shuffle(img_list)
        holdout_split = list_size - round(list_size*hold_out_size)
        print(holdout_split)
        holdout = img_list[holdout_split:]
        data = img_list[:holdout_split]
        return holdout, data
