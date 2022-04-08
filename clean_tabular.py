from cmath import nan
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
import json
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
import string
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import webcolors
import operator
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time


class TabularCleanse:
    """
    """
    def __init__(self, credentials_json, RDS_pass) -> None:
        # Get database connection credentials
        with open(os.path.join(os.getcwd(), credentials_json), mode='r') as f:
            database_dict = json.load(f)
        # Password for the database
        self.RDS_pass = RDS_pass
        # Create engine to connect to database
        self.engine = create_engine(f"{database_dict['DATABASE_TYPE']}+{database_dict['DBAPI']}://{database_dict['USER']}:{RDS_pass}@{database_dict['HOST']}:{database_dict['PORT']}/{database_dict['DATABASE']}")
        # Import the tables as DataFrame
    
    def get_data_table(self, table_name):
        data = pd.read_sql(table_name, self.engine)
        return data
    
    @staticmethod
    def remove_outlier_iqr(df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        return df_out
    
    @staticmethod
    def detect_outliers_iqr(data):
        outliers = []
        data = sorted(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        # print(q1, q3)
        IQR = q3-q1
        lwr_bound = q1-(1.5*IQR)
        upr_bound = q3+(1.5*IQR)
        # print(lwr_bound, upr_bound)
        for i in data: 
            if (i<lwr_bound or i>upr_bound):
                outliers.append(i)
        return outliers
    
    @staticmethod
    def detect_outliers_zscore(data):
        outliers = []
        thres = 3
        mean = np.mean(data)
        std = np.std(data)
        # print(mean, std)
        for i in data:
            z_score = (i-mean)/std
            if (np.abs(z_score) > thres):
                outliers.append(i)
        return outliers
    
    # def remove_stop_punctuation_num(df_field):
    #     #Remove stopwords, punctuation and numbers
    #     transformed_text = [remove_stopwords(x)\
    #         .translate(str.maketrans('','',string.punctuation))\
    #         .translate(str.maketrans('','',string.digits))\
    #         # for x in products_na_removed['product_name']]
    #         for x in df_field]
    #     return transformed_text

    # def stemSentence(sentence):
    #     porter = PorterStemmer()
    #     token_words = word_tokenize(sentence)
    #     stem_sentence = [porter.stem(word) for word in token_words]
    #     return ' '.join(stem_sentence)
    #     # text3 = pd.Series([stemSentence(x) for x in transformed_text])

    @staticmethod
    def data_profile_report(data, output_filename):
        prof = ProfileReport(data)
        prof.to_file(output_file= output_filename)

    @staticmethod
    def remove_rows_conditonal(data, field_names, condition, operation= '!='):
        # need to chekc is this gives same values as products.loc[products['category'] != 'N/A']
        rel_ops = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne
        }
        operation = rel_ops[operation]
        new_data = data.loc[operation(data[field_names], condition)]
        return new_data
    
    @staticmethod
    def split_df_value(data, field_to_split, split_by, new_field_names):
        data[new_field_names] = data[field_to_split].str.split(split_by, expand=True)
        return data

    @staticmethod
    def dynamic_split_df_value(data, field_to_split, split_by, expand=True, prefix=None, fill_na= None):
        max_categories = data[field_to_split].str.count(split_by).max() 
        max_categories += 1
        if fill_na == None:
            fill_na = pd.NA
        split_data = data[field_to_split].str.split(split_by, expand=expand).reindex(range(max_categories), axis=1).add_prefix(prefix).fillna(fill_na)
        split_data = pd.concat([data, split_data], axis=1)
        return split_data

    @staticmethod
    def manual_split_df_value(data, new_field_names, field_to_split, split_by, split_amount=1):
        data[new_field_names] = data[field_to_split].str.split(split_by, expand=True).reindex(range(split_amount), axis=1)
        return data
    
    @staticmethod
    def convert_str_float(data, new_field_name, field_to_convert, replace_value):
        data[new_field_name] = data[field_to_convert].replace(replace_value, '', regex=True).astype(float)
        return data
        
    @staticmethod
    def geocode_locations(data, field_to_geocode):
        """
        Uses the geocode library to get the long and lat information from
        the location field of the df
        Args:
            None
        Returns:
            None
        """
        locator = Nominatim(user_agent='aicore_cleaner')
        geocode = RateLimiter(locator.geocode,
                                   min_delay_seconds=0.2,
                                   return_value_on_exception=None)
        data['geocode'] = data[field_to_geocode].apply(geocode)
        data['long_lat'] = data['geocode'].apply(
            lambda loc: tuple(loc.point) if loc else None)
        return data
    
    @staticmethod
    def multiple_dummy_encoder(data, field_list, prefix_list):
        field_count = 0
        for field in field_list:
            encoded_df = pd.get_dummies(data[field], drop_first=True, prefix=prefix_list[field_count])
            data.drop([field_list[field_count]], axis=1, inplace=True)
            data = pd.concat([data, encoded_df], axis=1)
            field_count += 1
        return data

if __name__ == "__main__":
    cleanse = TabularCleanse('fb_marketplace_conn.json', 'aicore2022!')
    products = cleanse.get_data_table('products')
    # cleanse.data_profile_report(products, 'products.html')
    cleaning_products = cleanse.remove_rows_conditonal(products, 'category', 'N/A', '!=')
    cleaning_products = cleanse.split_df_value(cleaning_products, 'location', ',', ['local_area', 'city'])
    cleaning_products = cleanse.convert_str_float(cleaning_products, 'price_gbp', 'price', '[\£,]')
    cleaning_products = cleanse.remove_rows_conditonal(cleaning_products, 'price_gbp', 399000.00, '<')
    cleaning_products = cleanse.dynamic_split_df_value(cleaning_products, 'category', '/', True, 'sub_cat_')
    cleaning_products = cleanse.manual_split_df_value(cleaning_products, 'product_name', 'product_name', '|', 1)
    cleaning_products.dropna(axis='rows', subset='city')
    cleaning_products.sort_values(by='city',ascending=False)
    cleaning_products = cleaning_products.dropna(subset=['city'])
    geocode_start= time.perf_counter()
    print(f'Starting to geocode at {geocode_start}')
    cleaning_products = cleanse.geocode_locations(cleaning_products, 'location')
    geocode_end= time.perf_counter()
    print(f'Finished to geocode at {geocode_end}')
    cleaning_products.drop(['price','location', 'create_time', 'category', 'sub_cat_1','sub_cat_2', 'sub_cat_3', 'sub_cat_4', 'page_id'], axis=1, inplace=True)
    # cleaning_products.info()
    # Encoding
    sub_cat_0_one = pd.get_dummies(cleaning_products['sub_cat_0'], drop_first=True, prefix='sub_cat')
    # cleaning_products = cleanse.multiple_dummy_encoder(cleaning_products, ['sub_cat_0', 'city'], [None, None])
    cleaning_products.info()
    # Think about changing datatype to catgeories

    cleaning_products.to_pickle('cleaned_products.pkl')
