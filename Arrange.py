import numpy as np
import datetime
from pandas as pd
# from datetime import datetime
# from math import sin, cos, pi
#

 from classifier import load
#
# def arrange():
#     data = load()
#     return data['Latitude', 'Longitude', 'Datetime']
#
#
# days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#
# def sin_cos(n):
#     theta = 2 * pi * n
#     return (sin(theta), cos(theta))
#
# def get_cycles(d):
#     '''
#     Get the cyclic properties of a datetime,
#     represented as points on the unit circle.
#     :returns
#     dictionary of sine and cosine tuples
#     '''
#     month = d.month - 1
#     day = d.day - 1
#     return {
#         'month': sin_cos(month / 12),
#         'day': sin_cos(day / days_in_month[month]),
#         'weekday': sin_cos(d.weekday() / 7),
#         'hour': sin_cos(d.hour / 24),
#         'minute': sin_cos(d.minute / 60),
#         'second': sin_cos(d.second / 60)
#     }
#
#
# get_cycles(datetime(2018, 6, 3, 16, 51, 53))
# # {
# #     'month': (0.49999999999999994, -0.8660254037844387),
# #     'day': (0.40673664307580015, 0.9135454576426009),
# #     'weekday': (-0.7818314824680299, 0.6234898018587334),
# #     'hour': (-0.8660254037844384, -0.5000000000000004),
# #     'minute': (-0.8090169943749476, 0.5877852522924729),
# #     'second': (-0.6691306063588588, 0.7431448254773937)
# # }

def arrange():
    data = pd.read_csv('test_dataset_crimes.csv')
    return data['Latitude', 'Longitude', 'Datetime']

def time_to_minute(data):
    data['Datetime'] = data['datetime'].hour * 60 + data['datetime'].minute
    return data

def all_crimes(data):
    time_to_minute(data)
    return data

def crimes_by_weekday(data, date):
    time_to_minute(data)
    d

