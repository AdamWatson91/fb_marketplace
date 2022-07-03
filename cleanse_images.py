import os
import pandas as pd
from tqdm import tqdm
from image_cleanser import ImageCleanse
from numpy import asarray

img_data = pd.read_csv('Images.csv', index_col=0)
img_data.drop(['bucket_link', 'image_ref', 'create_time'], axis=1, inplace=True)
img_data.rename(columns={'id': 'img_id'}, inplace=True)
images_with_products = list(set(img_data['img_id'].to_list()))

download_image_directory = os.path.join(os.getcwd(), 'images_fb/images/')
upload_image_directory = os.path.join(os.getcwd(), 'images_fb/clean_images/')
cleanse = ImageCleanse(download_image_directory, upload_image_directory)
resized_images, image_names = cleanse.get_images(download_image_directory, 256, 'RGB')
holdout, images = cleanse.create_holdout(resized_images, 0.05)

for i, img in enumerate(tqdm(holdout, desc='Cleaning images...')):
    file = image_names[i]
    cleanse.save_images(cleanse.upload, 'test/original_256', img, file)

img_list = []
img_ids = []
product_ids = []

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


img_dict = {
    'img_id': img_ids,
    'img_array': img_list,
    'product_id': product_ids
}

img_tabular = pd.DataFrame(img_dict, columns=['img_id', 'img_array', 'product_id'])
pd.to_pickle(img_tabular, 'cleansed_img_tabular')
