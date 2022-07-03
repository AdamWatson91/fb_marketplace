import pandas as pd
import numpy as np

products = pd.read_pickle('products_cleansed.pkl')
img_tabular = pd.read_pickle('cleansed_img_tabular')

# Check which products to keep based on whther they have images
products_to_keep = list(set(img_tabular['product_id']))
products = products[products['product_id'].isin(products_to_keep)]

# Merge the two
product_images = products.merge(img_tabular, how='inner', left_on='product_id', right_on='product_id', validate='one_to_many')
# Ouput data for colab(.npy) and vsc (.pkl)
pd.to_pickle(product_images, 'cleansed_product_images.pkl')
product_images_np = product_images.to_numpy()
np.save(product_images_np, 'cleansed_product_images.npy')