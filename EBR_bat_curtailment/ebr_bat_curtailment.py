#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import timedelta, datetime
import numpy as np


# In[2]:


def get_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine='pyarrow')
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format=' %d/%m/%Y %H:%M')
    df.set_index(['TimeStamp', 'Name'], inplace=True)
    df.dropna(inplace=True)
    return df


# In[3]:


def get_sun_dataframe(path: str) -> pd.DataFrame:
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
    df = pd.read_csv(path, engine='pyarrow')
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format=' %d/%m/%Y %H:%M')
    df.set_index(['TimeStamp'], inplace=True)
    df.dropna(inplace=True)
    return df


# In[5]:


def add_sun_info(general_df: pd.DataFrame, sun_df: pd.DataFrame) -> pd.DataFrame:
    general_df["Day_Month"] = general_df.index.get_level_values(0).strftime('%m-%d')
    sun_df["Day_Month"] = sun_df.index.get_level_values(0).strftime('%m-%d')
    return general_df.reset_index().merge(sun_df, on='Day_Month', how='inner').set_index(['TimeStamp', 'Name']).drop(
        columns=['Day_Month'])


# In[6]:


def add_loss_column(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    df['In Timeframe'] = df.index.get_level_values(0).to_series().between(start_date, end_date).to_numpy()
    return df


# In[7]:


def add_night_column(df_general: pd.DataFrame) -> pd.DataFrame:
    dt = pd.Series(df_general.index.get_level_values(0))
    dt_from = dt - pd.Timedelta(minutes=10)

    dt = dt.dt.time.to_numpy()
    dt_from = dt_from.dt.time.to_numpy()

    start_curt = df_general['Start Curtailment'].to_numpy()
    end_curt = df_general['End Curtailment'].to_numpy()

    mask = np.where(start_curt > end_curt,
                    ((start_curt <= dt) | (dt_from <= end_curt)),
                    ((start_curt <= dt) & (dt_from <= end_curt)))

    df_general['Night'] = mask
    return df_general


# In[8]:


def add_wind_column(df_general: pd.DataFrame, wind_thres: float) -> pd.DataFrame:
    df_general['Below Wind'] = df_general['WindSpeed'] < wind_thres
    return df_general


# In[9]:


def add_temps_columns(df_general: pd.DataFrame, df_temp: pd.DataFrame, low: float = 0,
                      high: float = 40) -> pd.DataFrame:
    merged_df = pd.merge(df_general, df_temp, left_index=True, right_index=True, how='inner')
    merged_df['In Thres Temp'] = (merged_df['Temp'] >= low) & (merged_df['Temp'] <= high)
    return merged_df[
        ['WindSpeed', 'ActivePower', 'Energy', 'Temp', 'Start Curtailment', 'End Curtailment', 'In Timeframe', 'Night',
         'Below Wind', 'In Thres Temp']]


# In[ ]:


def calculate_loss(processed_df: pd.DataFrame, standby_loss: float = 20.0) -> (pd.DataFrame, pd.DataFrame):
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
    test = add_wind_column(test, wind_thres=5)
    test = add_temps_columns(test, temps_test)
    test_lost, test_produced = calculate_loss(test)
    test_lost['Energy'].groupby(level=1).aggregate('sum')
    test_produced['Energy'].groupby(level=1).aggregate('sum')


# In[ ]:


if __name__ == '__main__':
    main()
