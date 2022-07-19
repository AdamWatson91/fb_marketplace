import os
import pandas as pd
from tqdm import tqdm
from cleanse_images import ImageCleanse
from numpy import asarray

# Read in image details data from csv
img_data = pd.read_csv('Images.csv', index_col=0)
# Drop irrelevant fields
img_data.drop(['bucket_link', 'image_ref', 'create_time'], axis=1, inplace=True)
# Rename identifies to reduce conflict when later merging with products data
img_data.rename(columns={'id': 'img_id'}, inplace=True)
# Create list of uid's for images in csv. Some images are not included in this
# dataset and therefore will be removed as there is no product data for them.
images_with_products = list(set(img_data['img_id'].to_list()))

# Download the source images
download_image_directory = os.path.join(os.getcwd(), 'images_fb/images/')
# Specify the upload location once cleansed
upload_image_directory = os.path.join(os.getcwd(), 'images_fb/clean_images/')
# Intialise image cleansing class 
cleanse = ImageCleanse(download_image_directory, upload_image_directory)
# Resize images and ensure same mode
resized_images, image_names = cleanse.get_images(download_image_directory, 256, 'RGB')
# Create a holdout set for testing purposes
holdout, images = cleanse.create_holdout(resized_images, 0.05)

# Save images for testing from holdout
for i, img in enumerate(tqdm(holdout, desc='Cleaning images...')):
    file = image_names[i]
    cleanse.save_images(cleanse.upload, 'test/original_256', img, file)

# Initialise the empy lists which will hold image data
img_list = []
img_ids = []
product_ids = []

# Clean and save images for model development
# Store the image detial data in tabular format
for i, img in enumerate(tqdm(images, desc='Cleaning images...')):
    file = image_names[i]
    img_id = file.replace('.jpg', '')
    if img_id in images_with_products:
        img_ids.append(img_id)
        cleanse.save_images(cleanse.upload, 'data/original_256', img, file)
        img_array = asarray(img)
        img_list.append(img_array)
        product_id = img_data.set_index('img_id').loc[img_id, 'product_id']
        product_ids.append(product_id)

# Create dictionary with relevant image detail data
img_dict = {
    'img_id': img_ids,
    'img_array': img_list,
    'product_id': product_ids
}

# Create DataFrame format of image detail data and output to pickle
img_tabular = pd.DataFrame(img_dict, columns=['img_id', 'img_array', 'product_id'])
pd.to_pickle(img_tabular, 'cleansed_img_tabular.pkl')
