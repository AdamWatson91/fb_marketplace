import pandas as pd
from tabular_cleanser import TabularCleanse

cleanse = TabularCleanse('fb_marketplace_conn.json', 'aicore2022!')

cleaning_products = pd.read_csv('Products.csv', index_col=0, lineterminator='\n')
cleaning_products = cleanse.remove_rows_conditonal(cleaning_products, 'category', 'N/A', '!=')
cleaning_products = cleanse.split_df_value(cleaning_products, 'location', ',', ['local_area', 'city'])
cleaning_products = cleanse.convert_str_float(cleaning_products, 'price_gbp', 'price', '[\£,]')
cleaning_products = cleanse.remove_rows_conditonal(cleaning_products, 'price_gbp', 1000.00, '<')
cleaning_products = cleanse.dynamic_split_df_value(cleaning_products, 'category', '/', True, 'sub_cat_')
cleaning_products = cleanse.manual_split_df_value(cleaning_products, 'product_name', 'product_name', '|', 1)
cleaning_products = cleaning_products.dropna(subset=['city'])
cleaning_products.rename(columns={'id': 'product_id'}, inplace=True)
pd.to_pickle(cleaning_products, 'products_cleansed.pkl')
print(cleaning_products.head(5))
print(cleaning_products.info())
