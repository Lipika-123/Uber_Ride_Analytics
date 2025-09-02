# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/uber-ride-analytics-dashboard'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv("/kaggle/input/uber-ride-analytics-dashboard/ncr_ride_bookings.csv")
df.head(5)

# PERFORMING INITIAL ANALYSIS OF DATASET
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

df.isna().sum()
numeric_col = [col for col in df.columns if df[col].dtype != 'object']
numeric_col

import seaborn as sns

sns.heatmap(df[numeric_col].corr())
#1. From this we can assume that booking value depends on factors like ride distance , driver ratings and customer rating
#2. Also depended on vtat and ctat
#3.avg vtat and ctat are linked with each other

# PERFORM MISSING VALUE TREATMENT
## for numeric columns that have float values,replace the missing values by mean
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

target_columns = [
    'Avg VTAT',
    'Avg CTAT',
    'Booking Value',
    'Ride Distance',
    'Driver Ratings',
    'Customer Rating'
]

df[target_columns] = imp_mean.fit_transform(df[target_columns])

## for the columns having columns in the format of 1/0 replace the missing values as 0
df["Cancelled Rides by Customer"] = df["Cancelled Rides by Customer"].replace(np.nan , 0)
df["Cancelled Rides by Customer"].value_counts()
df["Cancelled Rides by Driver"] = df["Cancelled Rides by Driver"].replace(np.nan , 0)
df["Cancelled Rides by Driver"].value_counts()
df["Incomplete Rides"] = df["Incomplete Rides"].replace(np.nan,0)
df["Incomplete Rides"].value_counts()

## for Payment column replace the Nan to most frequent count of payment method
from sklearn.impute import SimpleImputer
imp_cat = SimpleImputer(strategy='most_frequent')
# fit_transform returns 2D array, so flattening it using [:, 0]
df["Payment Method"] = imp_cat.fit_transform(df[["Payment Method"]])[:, 0]

## for columns containing reasons, keep the NA values as Unknown
df['Reason for cancelling by Customer'].fillna('Unknown', inplace=True)
df["Driver Cancellation Reason"].fillna('Unknown', inplace=True)
df['Incomplete Rides Reason'].fillna('Unknown', inplace=True)

# If Date and Time are separate columns
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time

# Combine Date and Time into a single datetime column (optional)
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

#ENCODING THE CATEGORICAL COLUMN FOR BETTER ANALYSIS
df["Booking Status"].unique()

df["Booking Status_c"] = df["Booking Status"].map({
                                                    'No Driver Found' : 0,
                                                    'Incomplete':1,
                                                    'Completed':2,
                                                    'Cancelled by Driver':3,
                                                    'Cancelled by Customer':4
                                                })
df["Vehicle Type"].unique()

df["Vehicle Type_c"] = df["Vehicle Type"].map({
    'eBike':0,
    'Go Sedan': 1,
    'Auto': 2,
    'Premier Sedan': 3,
    'Bike':4,
    'Go Mini':5,
    'Uber XL':6  
})
df["Reason for cancelling by Customer"].unique()
df["Reason for cancelling by Customer_c"] = df["Reason for cancelling by Customer"].map({
    'Unknown':0,
    'Driver is not moving towards pickup location':1,
    'Driver asked to cancel':2,
    'AC is not working':3,
    'Change of plans':4,
    'Wrong Address':5 
})
df["Driver Cancellation Reason"].unique()
df["Driver Cancellation Reason_c"] = df["Driver Cancellation Reason"].map({
    'Unknown':0,
    'Personal & Car related issues':1,
    'Customer related issue':2,
    'More than permitted people in there':3,
    'The customer was coughing/sick':4
})
df["Incomplete Rides Reason"].unique()
df["Incomplete Rides Reason_c"] = df["Incomplete Rides Reason"].map({
    'Unknown':0,
    'Vehicle Breakdown':1,
    'Other Issue':2,
    'Customer Demand':3
})
df["Payment Method"].unique()
df["Payment Method_c"] = df["Payment Method"].map({
    'UPI':0,
    'Debit Card':1,
    'Cash':2,
    'Uber Wallet':3,
    'Credit Card':4
})
df.info()
df.to_csv('Uber_analytics_processed',index = True)


