#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime

import numpy as np
import pandas as pd


# In[2]:


def get_dataframe(path: str) -> pd.DataFrame:
    """
    get_dataframe(path: str) -> pd.DataFrame

    This function reads a CSV file located at the given path and returns a pandas DataFrame. It performs the following operations on the DataFrame:
    - Converts the 'TimeStamp' column to datetime format using the specified format.
    - Sets the index of the DataFrame to 'TimeStamp' and 'Name' columns.
    - Drops rows with missing values.

    Parameters:
    - path (str): The file path of the CSV file.

    Returns:
    - pd.DataFrame: The resulting DataFrame after performing the operations.

    Example:
    >>> get_dataframe('data.csv')
                         Value
    TimeStamp           Name
    2021-01-01 09:00:00 A       10
    2021-01-01 09:30:00 B        5
    2021-01-01 10:00:00 C       15
    2021-01-01 10:30:00 D       20

    Note: This function assumes that the CSV file is formatted correctly and contains the required columns.
    """
    df = pd.read_csv(path, engine='pyarrow')
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format=' %d/%m/%Y %H:%M')
    df.set_index(['TimeStamp', 'Name'], inplace=True)
    df.dropna(inplace=True)
    return df


# In[3]:


def get_sun_dataframe(path: str) -> pd.DataFrame:
    """
    Reads a CSV file from the provided path and returns a pandas DataFrame.

    Parameters:
    - path: str - The path of the CSV file to be read.

    Returns:
    - pd.DataFrame - The DataFrame generated from the CSV file.

    Example Usage:
    path = 'data/sun_data.csv'
    df = get_sun_dataframe(path)
    """
    df = pd.read_csv(path, engine='pyarrow',
                     usecols=['Date', 'End Curtailment', 'Start Curtailment'],
                     dtype={'Date': 'datetime64[ns]',
                            'End Curtailment': 'str',
                            'Start Curtailment': 'str'})

    df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%y")
    df.set_index('Date', inplace=True)
    for column in ['Start Curtailment', 'End Curtailment']:
        df[column] = pd.to_datetime(df[column], format='%H:%M', errors='coerce').fillna(
            pd.to_datetime(df[column], format='%H:%M:%S', errors='coerce')).dt.time

    return df


# In[4]:


def get_temp_dataframe(path: str) -> pd.DataFrame:
    """

    Parameters:
    -----------
    path : str
        The file path of the temperature CSV file to be read.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the temperatures from the CSV file.

    """
    df = pd.read_csv(path, engine='pyarrow')
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format=' %d/%m/%Y %H:%M')
    df.set_index(['TimeStamp'], inplace=True)
    df.dropna(inplace=True)
    return df


# In[5]:


def add_sun_info(general_df: pd.DataFrame, sun_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sun information to a general dataframe.

    Parameters:
        general_df (pd.DataFrame): The general dataframe to which sun information is added.
        sun_df (pd.DataFrame): The dataframe containing sun information.

    Returns:
        pd.DataFrame: The modified general dataframe with sun information added.

    Example:
        >>> general = pd.DataFrame({'TimeStamp': ['2021-01-01 12:00:00', '2021-01-02 12:00:00'],
                                   'Name': ['A', 'B']})
        >>> sun = pd.DataFrame({'TimeStamp': ['2021-01-01 12:00:00', '2021-01-02 12:00:00'],
                                'Day_Month': ['01-01', '01-02'],
                                'Sunrise': ['06:00:00', '06:30:00'],
                                'Sunset': ['18:00:00', '18:30:00']})
        >>> add_sun_info(general, sun)
           TimeStamp            Name  Sunrise   Sunset
        0 2021-01-01 12:00:00    A   06:00:00 18:00:00
        1 2021-01-02 12:00:00    B   06:30:00 18:30:00
    """
    general_df["Day_Month"] = general_df.index.get_level_values(0).strftime('%m-%d')
    sun_df["Day_Month"] = sun_df.index.get_level_values(0).strftime('%m-%d')
    return general_df.reset_index().merge(sun_df, on='Day_Month', how='inner').set_index(['TimeStamp', 'Name']).drop(
        columns=['Day_Month'])


# In[6]:


def add_loss_column(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """

    Add Loss Column

    Adds a new column to the given DataFrame indicating whether each row falls within a specified timeframe.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - start_date (str): The start date of the timeframe in the format 'YYYY-MM-DD'.
    - end_date (str): The end date of the timeframe in the format 'YYYY-MM-DD'.

    Returns:
    - pd.DataFrame: The modified DataFrame with the additional 'In Timeframe' column.

    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    df['In Timeframe'] = df.index.get_level_values(0).to_series().between(start_date, end_date).to_numpy()
    return df


# In[7]:


def add_night_column(df_general: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Night' column to the given DataFrame.

    Parameters:
        df_general (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'Night' column added.
    """
    dt = pd.Series(df_general.index.get_level_values(0))

    dt = dt.dt.time.to_numpy()

    start_curt = df_general['Start Curtailment'].to_numpy()
    end_curt = df_general['End Curtailment'].to_numpy()

    mask = np.where(start_curt > end_curt,
                    ((start_curt <= dt) | (dt <= end_curt)),
                    ((start_curt <= dt) & (dt <= end_curt)))

    df_general['Night'] = mask
    return df_general


# In[8]:


def add_wind_column(df_general: pd.DataFrame, wt_T01: float = 5.0, wt_T02: float = 5.0, wt_T03: float = 5.0,
                    wt_T05: float = 5.0, wt_T06: float = 5.0, wt_T09: float = 5.0) -> pd.DataFrame:
    """

    Add Wind Column to DataFrame

    Adds a new column to a given DataFrame indicating whether the wind speed for each row is below a certain threshold.

    Parameters:
    - df_general (pd.DataFrame): The DataFrame to which the wind column will be added.
    - wt_T01 (float, optional): The wind speed threshold for T01. Default is 5.0.
    - wt_T02 (float, optional): The wind speed threshold for T02. Default is 5.0.
    - wt_T03 (float, optional): The wind speed threshold for T03. Default is 5.0.
    - wt_T05 (float, optional): The wind speed threshold for T05. Default is 5.0.
    - wt_T06 (float, optional): The wind speed threshold for T06. Default is 5.0.
    - wt_T09 (float, optional): The wind speed threshold for T09. Default is 5.0.

    Returns:
    - pd.DataFrame: The updated DataFrame with the additional 'Below Wind' column.

    """
    wind_thresholds = {
        'T01': wt_T01,
        'T02': wt_T02,
        'T03': wt_T03,
        'T05': wt_T05,
        'T06': wt_T06,
        'T09': wt_T09,
    }
    df_general['Below Wind'] = df_general.apply(
        lambda row: row['WindSpeed'] < wind_thresholds[row.name[1]], axis=1)
    return df_general


# In[9]:


def add_temps_columns(df_general: pd.DataFrame, df_temp: pd.DataFrame, low: float = 0,
                      high: float = 40) -> pd.DataFrame:
    """
    Add temperature columns to a DataFrame

    This method takes two DataFrames - df_general and df_temp, and adds temperature-related columns to df_general DataFrame based on the temperature values in df_temp DataFrame. The added
    * columns include:

    1. In Thres Temp: A boolean column indicating whether the temperature falls within the specified range [low, high].

    Parameters:
    - df_general : pd.DataFrame
        The general DataFrame to which the temperature columns will be added.
    - df_temp : pd.DataFrame
        The DataFrame containing temperature data that will be used to add the columns.
    - low : float, optional
        The lower bound of the temperature range. Defaults to 0.
    - high : float, optional
        The upper bound of the temperature range. Defaults to 40.

    Returns:
    - pd.DataFrame
        The merged DataFrame containing the original columns from df_general and the added temperature columns.

    Example Usage:
    df_general = pd.DataFrame(...)
    df_temp = pd.DataFrame(...)
    merged_df = add_temps_columns(df_general, df_temp, low=10, high=30)

    """
    merged_df = pd.merge(df_general, df_temp, left_index=True, right_index=True, how='inner')
    merged_df['In Thres Temp'] = (merged_df['Temp'] >= low) & (merged_df['Temp'] <= high)
    return merged_df[
        ['WindSpeed', 'ActivePower', 'Energy', 'Temp', 'Start Curtailment', 'End Curtailment', 'In Timeframe', 'Night',
         'Below Wind', 'In Thres Temp']]


# In[ ]:


def calculate_loss(processed_df: pd.DataFrame, standby_loss: float = 20.0) -> (pd.DataFrame, pd.DataFrame):
    """

    This method, calculate_loss, calculates the energy loss and energy produced from a processed DataFrame.

    Parameters:
    - processed_df: A pandas DataFrame containing the processed data.
    - standby_loss: A float representing the standby loss value. Default value is set to 20.0.

    Returns:
    - lost_df: A pandas DataFrame containing the rows from processed_df that meet the condition for energy loss.
        Columns in lost_df include 'Energy Pulled' that is set to the standby_loss value.
    - produced_df: A pandas DataFrame containing the rows from processed_df that meet the condition for energy production.

    Example usage:
    processed_df = pd.DataFrame(...)
    lost_df, produced_df = calculate_loss(processed_df, standby_loss=30.0)
    """
    processed_df = processed_df[(processed_df['Energy'] > -40) & (processed_df['Energy'] < 700)]
    lost_df = processed_df[
        processed_df['Night'] & processed_df['Below Wind'] & processed_df['In Timeframe'] & processed_df[
            'In Thres Temp']]
    produced_df = processed_df[
        ~(processed_df['Night'] & processed_df['Below Wind'] & processed_df['In Thres Temp']) & processed_df[
            'In Timeframe']]
    lost_df.loc[:, 'Energy Pulled'] = standby_loss
    return lost_df, produced_df


# In[29]:


def main():
    test = get_dataframe(r'2023.csv')
    date_test = get_sun_dataframe(r'EBR_sun.csv')
    temps_test = get_temp_dataframe(r'2023_temp.csv')
    test = add_loss_column(test, "2022-05-01", "2022-06-01")
    test = add_sun_info(test, date_test)
    test = add_night_column(test)
    test = add_wind_column(test, wt_T01=5, wt_T02=5, wt_T03=5, wt_T05=5, wt_T06=5, wt_T09=5)
    test = add_temps_columns(test, temps_test)
    test_lost, test_produced = calculate_loss(test)
    test_lost['Energy'].groupby(level=1).aggregate('sum')
    test_produced['Energy'].groupby(level=1).aggregate('sum')


# In[ ]:


if __name__ == '__main__':
    main()
