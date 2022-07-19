import pandas as pd
from clean_tabular import TabularCleanse

# Initialise TabularCleanse
cleanse = TabularCleanse('fb_marketplace_conn.json', 'aicore2022!')

# Import product data from csv
cleaning_products = pd.read_csv('Products.csv', index_col=0, lineterminator='\n')
# Remove null values
cleaning_products = cleanse.remove_rows_conditonal(cleaning_products, 'category', 'N/A', '!=')
# Create granular location fields
cleaning_products = cleanse.split_df_value(cleaning_products, 'location', ',', ['local_area', 'city'])
# Create price field that is numeric
cleaning_products = cleanse.convert_str_float(cleaning_products, 'price_gbp', 'price', '[\£,]')
# Remove outliers in the price field
cleaning_products = cleanse.remove_rows_conditonal(cleaning_products, 'price_gbp', 1000.00, '<')
# Split hierarchical category field into seerate category fields
cleaning_products = cleanse.dynamic_split_df_value(cleaning_products, 'category', '/', True, 'sub_cat_')
# Keep only the first string from product name
cleaning_products = cleanse.manual_split_df_value(cleaning_products, 'product_name', 'product_name', '|', 1)
# Drop null values from city
cleaning_products = cleaning_products.dropna(subset=['city'])
# Rename uid field for later merge
cleaning_products.rename(columns={'id': 'product_id'}, inplace=True)
# Output to pickle for future developments
pd.to_pickle(cleaning_products, 'products_cleansed.pkl')
