import math
import pandas as pd
import numpy as np


def str2bool(str_b):
    str_b = str(str_b).lower()
    if ((str_b == 'nan') or (str_b == 'none') or (str_b == '')):
        return None
    else:
        return str_b in ("true", "t", "1", "yes")  # different optiont to represent True as a string => change it

# Example:
# a=str2bool('')
# print(a)

#---------------------------------------
def float_to_int(num):
    if (math.isnan(num)):
        return float('NaN')
    else:
        return int(num)

# TEST
# test = float_to_int( float('NaN') )
# test2 = float_to_int( 9.6)off

# print(test, test2)
#---------------------------------------

# # s_hex - string that represents number in hexa => ex: 'dfdfdf'
# def str_to_hexa(s_hex):
#     return hex(int(s_hex, base=16))[2:]   # returns str that represents hexa without '0x' => {'0xdfdfdf' --> 'dfdfdf'}

# t_hex = str_to_hexa(s_hex='ffd201')
# print(type(t_hex))
# t_hex
#---------------------------------------

# convert time convention
def convert_to_datetime(df, cols):
    for col in cols:
        df['%s' % col] = pd.to_datetime(df['%s' % col], unit='s')
    return df

# convert Series of types String or Int to Timestamp:
def convert_Series_to_Timestamp(df, column):
    return pd.to_datetime(df[column].astype(str)).values
#---------------------------------------

# # create timeline features => CHANGE COLUMNS NAMES!!!
# def create_features_timeline(df):
#     df['duration_to_deadline_in_days'] = (df.loc[:, 'deadline'] - df.loc[:, 'launched_at']).apply(lambda l: l.days)
#     df['month_start'] = df.loc[:, 'launched_at'].apply(lambda l: l.month)
#     df['year_start'] = df.loc[:, 'launched_at'].apply(lambda l: l.year)
#     df['dayofyear_start'] = df.loc[:, 'launched_at'].apply(lambda l: l.dayofyear)
#     df['month_deadline'] = df.loc[:, 'deadline'].apply(lambda l: l.month)
#     df['year_deadline'] = df.loc[:, 'deadline'].apply(lambda l: l.year)
#     df['dayofyear_deadline'] = df.loc[:, 'deadline'].apply(lambda l: l.dayofyear)
#     return df
