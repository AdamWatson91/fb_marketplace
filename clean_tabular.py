import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
import json
from pandas_profiling import ProfileReport
import string
import numpy as np
import operator
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from typing import Union


class TabularCleanse:
    """
    This class provides the common functions used to cleanse a dataset that is
    in a tabular structured format. It can be used to access data from a AWS
    RDS database or intialised to use it's methods.

    Args:
        credentials_json (json): json file with the connection credtials to
                                    the required AWS RDS
        RDS_pass (str): The password for the AWS RDS
    """
    def __init__(self, credentials_json: json, RDS_pass: str) -> None:
        """See help(TabularCleanse) for accurate signature"""
        # Get database connection credentials
        with open(os.path.join(os.getcwd(), credentials_json), mode='r') as f:
            database_dict = json.load(f)
        # Password for the database
        self.RDS_pass = RDS_pass
        # Create engine to connect to database
        self.engine = create_engine(f"{database_dict['DATABASE_TYPE']}+{database_dict['DBAPI']}://{database_dict['USER']}:{RDS_pass}@{database_dict['HOST']}:{database_dict['PORT']}/{database_dict['DATABASE']}")

    def get_data_table(self, table_name):
        data = pd.read_sql(table_name, self.engine)
        return data

    @staticmethod
    def remove_outlier_iqr(df_in: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Removes values from a DataFrame that are considered outliers based on
        the inter quartile range (iqr).

        Args:
            df_in (pd.DataFrame): The DataFrame with all the data
            col_name (str): The name of the colunm to use for dtermine iqr and
                                assessing values to remove.

        Returns:
            df_out (pd.DataFrame): The new DataFrame with outlier rows removed.
        """
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        # Interquartile range
        iqr = q3-q1 
        fence_low = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        return df_out

    @staticmethod
    def detect_outliers_iqr(data: list) -> list:
        """
        Calculates the inter quartile range (iqr) from list of values. Outputs any values that are
        considered outliers based on this iqr into a new list.

        Args:
            data (list): list of numerical values to calculate iqr
                            and assess against this score.

        Returns:
            outliers (list): List of values from original list that could be
                                considered outliers.
        """
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
            if (i < lwr_bound or i > upr_bound):
                outliers.append(i)
        return outliers

    @staticmethod
    def detect_outliers_zscore(data: list) -> list:
        """
        Calculates the zscore from list of values. Outputs any values that are
        considered outliers based on this zscore into a new list.

        Args:
            data (list): list of numerical values to calculate zscore
                            and assess against this score.

        Returns:
            outliers (list): List of values from original list that could be
                                considered outliers.
        """
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

    @staticmethod
    def data_profile_report(data: pd.DataFrame, output_filename: str) -> None:
        """
        Creates a .html file with a data profile usinf the ProfileReport class
        from pandas.profiling.

        Args:
            data (pd.DataFrame): The DataFrame with all the data.
            output_filename (str): The desired name of the file.

        Returns:
            None
        """
        prof = ProfileReport(data)
        prof.to_file(output_file=output_filename)

    @staticmethod
    def remove_rows_conditonal(data: pd.DataFrame, field_names: str, condition: Union[str, int, float], operation: str = '!=') -> pd.DataFrame:
        """
        Removes rows from a DataFrame if values in a specified field meet a
        condition. The user can determien the condition and the logic operator
        based on set values.

        Args:
            data (pd.DataFrame): The DataFrame with all the data.
            field_names (str): The colunm name that the values will be assessed
                                for removal.
            condition (str or int or float): The value that will be that
                                field_names value will be assessed against
                                with the operation.
            operation (str): The logical operator to be used. Operation that
                                are available and their couterpart fucntion
                                are:
                                    '>': operator.gt,
                                    '<': operator.lt,
                                    '>=': operator.ge,
                                    '<=': operator.le,
                                    '==': operator.eq,
                                    '!=': operator.ne

        Returns:
            new_data (pd.DataFrame): The new dataframe with the rows that met the condition removed.
        """
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
    def split_df_value(data: pd.DataFrame, field_to_split: str, split_by: str, new_field_names: list) -> pd.DataFrame:
        """
        Splits a string field value based on a specified field once,
        into two new fields.

        Args:
            data (pd.DataFrame): The DataFrame with all the data.
            field_to_split (str): The colunm name that will be split.
            split_by (str): The value that will used to split.
            new_field_names (list): The two field names used for split values.

        Returns:
            data (pd.DataFrame): The new dataframe with the new fields.
        """
        data[new_field_names] = data[field_to_split].str.split(split_by, expand=True)
        return data

    @staticmethod
    def dynamic_split_df_value(data: pd.DataFrame, field_to_split: str, split_by: str, expand: bool = True, prefix: str = None, fill_na=None) -> pd.DataFrame:
        """
        Splits a string field value based on a specified field value into
        seperate DataFrame columns. It will calculate how many of the parse
        value to split exist and then split the number of times.

        Args:
            data (pd.DataFrame): The DataFrame with all the data.
            field_to_split (str): The colunm name that will be split.
            split_by (str): The value that will used to split.
            expand (bool): Expand the split strings into separate columns.
                                If True, return DataFrame/MultiIndex
                                expanding dimensionality.
                                If False, return Series/Index, containing
                                lists of strings.
            prefix (str): The prefix that will be used as the name. This
                                will be folunm by the index of the split.
            fill_na (str or None): The value that will be used to repalce
                                NaN values after splitting. If none then
                                pd.NA is used.

        Returns:
            split_data (pd.DataFrame): The new dataframe with the new fields.
        """
        max_categories = data[field_to_split].str.count(split_by).max()
        max_categories += 1
        if fill_na is None:
            fill_na = pd.NA
        split_data = data[field_to_split].str.split(split_by, expand=expand).reindex(range(max_categories), axis=1).add_prefix(prefix).fillna(fill_na)
        split_data = pd.concat([data, split_data], axis=1)
        return split_data

    @staticmethod
    def manual_split_df_value(data: pd.DataFrame, new_field_names: str, field_to_split : str, split_by: str, split_amount: int = 1) -> pd.DataFrame:
        """
        Splits a string field value based on a specified field value into
        seperate DataFrame columns.For example, splitting the string '123/A'
        by a '/', would result in two fields, one cotnaining ''123 and the
        other coontaining 'A'. Original field is NOT dropped. The user can
        specify the max number of splits to perform.

        Args:
            data (pd.DataFrame): The DataFrame with all the data.
            new_field_names (str): The column name desired for the new fields.
            field_to_split (str): The colunm name that will be split.
            split_value (str): The value that will used to split.
            split_amount (int): The max number of splits to perform. Remaining
                                    string will be added to the final column.

        Returns:
            data (pd.DataFrame): The original dataframe with the new fields.
        """
        data[new_field_names] = data[field_to_split].str.split(split_by, expand=True).reindex(range(split_amount), axis=1)
        return data

    @staticmethod
    def convert_str_float(data: pd.DataFrame, new_field_name: str, field_to_convert: str, replace_value: str) -> pd.DataFrame:
        """
        Converts a string field in a DataFrame to a float by removing user
        specified values. For example, converting a price field that includes
        the currency symbol. New field is added and the existing field is NOT
        dropped.

        Args:
            data (pd.DataFrame): The DataFrame with all the data.
            new_field_name (str): The column name desired for the new field.
            field_to_convert (str): The colunm name that will be converted.
            replace_value (str): The value that will be repalced to ''

        Returns:
            data (pd.DataFrame): The original dataframe with the new field.
        """
        data[new_field_name] = data[field_to_convert].replace(replace_value, '', regex=True).astype(float)
        return data

    @staticmethod
    def geocode_locations(data: pd.DataFrame, field_to_geocode: str) -> pd.DataFrame:
        """
        Uses the geocode library to get the long and lat information from
        the location field of the df.

        Args:
            data (pd.DataFrame): The dataframe with all the data including
                                    field to be geocoded.
            field_to_geocode (str): The column name in data that holds the
                                    location data to parse to the geocder.
        Returns:
            data (pd.DataFrame): The original dataframe with the geocoded
                                    data.
        """
        locator = Nominatim(user_agent='aicore_cleaner')
        geocode = RateLimiter(locator.geocode,
                                min_delay_seconds=0.2,
                                return_value_on_exception=None)
        data['geocode'] = data[field_to_geocode].apply(geocode)
        data['long_lat'] = data['geocode'].apply(
            lambda loc: tuple(loc.point) if loc else None)
        data[['long', 'lat', 'remove']] = pd.DataFrame(data['long_lat'].tolist(), index=data.index)
        data.drop(['remove', 'geocode', 'long_lat'], axis=1, inplace=True)
        return data

    @staticmethod
    def multiple_dummy_encoder(data: pd.DataFrame, field_list: list, prefix_list: list, drop_first: bool = True) -> pd.DataFrame:
        """
        This enables the user to perform one hot encoding or dummy
        encoding for singular or multiple fields in a pandas DataFrame
        based on the fields entered by the user.

        Args:
            data (pd.DataFrame): The dataframe with all the data including
                                    fields that will be encoded.
            field_list (list): List of field names from data that will be
                                    encoded.
            prefix_list (list): List containing the prefix to be added the
                                    field in the corresponding index from
                                    field_list
            drop_first (bool): True will perform dummy enocding and False will
                                    perform one hot encoding.

        Returns:
            data (pd.DataFrame): The original DataFrame with the encoding
                                    applied. Encoded fields are added to
                                    the end of the existing DataFramee
        """
        field_count = 0
        for field in field_list:
            encoded_df = pd.get_dummies(data[field],
                                        drop_first=drop_first,
                                        prefix=prefix_list[field_count])
            data.drop([field_list[field_count]], axis=1, inplace=True)
            data = pd.concat([data, encoded_df], axis=1)
            field_count += 1
        return data
