# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/uber-analytics-processed'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv("/kaggle/input/uber-analytics-processed/Uber_analytics_processed.csv")
print(df.head(5))

# FEATURE ENGINEERING

df.info()
# Getting some Datetime level columns added for analysis
df['DateTime'] =pd.to_datetime(df['DateTime'])
df['Pickup_hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day_name()
df['Month'] = df['DateTime'].dt.month_name()
df.head(5)

# Define some business metrics
df['Uber_commission'] = df['Booking Value']*0.25
df['fare_per_km'] = df['Booking Value']/df['Ride Distance']
df['driver_speed_eff']= df['Ride Distance']/df['Avg VTAT']
df['Wait time ratio'] = df['Avg CTAT']/df['Avg VTAT']

# DESCRIPTIVE & TIME SERIES ANALYSIS

# 1. Booking Status Overview
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Overall Booking Status Distribution
status_counts = df['Booking Status'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Overall Distribution of Ride Booking Status')
plt.axis('equal')
plt.show()

# Breakdown of Cancellations (Since Unknown is the maximum will remove that to see what are the other top reasons)
cancel_reasons_customer = df['Reason for cancelling by Customer'].replace('Unknown',np.nan).dropna().value_counts().head(5)
cancel_reasons_driver = df['Driver Cancellation Reason'].replace('Unknown',np.nan).dropna().value_counts().head(5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
cancel_reasons_customer.plot(kind='bar', ax=ax1, title='Top 5 Customer Cancellation Reasons')
cancel_reasons_driver.plot(kind='bar', ax=ax2, title='Top 5 Driver Cancellation Reasons')
plt.tight_layout()
plt.show()

# 2.Time Based Analysis
# Convert 'Date' to datetime and set as index for time series analysis
df['Date'] = pd.to_datetime(df['Date'])
df_time = df.set_index('Date')

# Daily Ride Volume (Completed, Cancelled, Incomplete)
daily_volume = df_time.resample('D')['Booking ID'].count()
daily_completed = df_time[df_time['Booking Status'] == 'Completed'].resample('D')['Booking ID'].count()
daily_cancelled = df_time[(df_time['Booking Status'] == 'Cancelled by Driver')| (df_time['Booking Status'] == 'Cancelled by Customer')].resample('D')['Booking ID'].count()

plt.figure(figsize=(14, 6))
daily_volume.plot(label='Total Bookings', alpha=0.7)
daily_completed.plot(label='Completed', alpha=0.7)
daily_cancelled.plot(label='Cancelled', alpha=0.7)
plt.title('Daily Ride Volume Trends')
plt.ylabel('Number of Rides')
plt.legend()
plt.show()

# Average Booking Value by Month
monthly_avg_value = df_time.resample('M')['Booking Value'].mean()
monthly_avg_value.plot(title='Average Monthly Booking Value', ylabel='Average Value ($)')
plt.show()

# Demand by Hour of Day
plt.figure(figsize=(12, 5))
df['Pickup_hour'].value_counts().sort_index().plot(kind='bar')
plt.title('Ride Demand by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=0)
plt.show()

# 3.Vehicle Type Analysis
vehicle_stats = df.groupby('Vehicle Type').agg({
    'Booking ID': 'count',
    'Booking Value':'mean',
    'Ride Distance' : 'mean',
    'Driver Ratings' : 'mean',
    'Wait time ratio' : 'mean'   
}).rename(columns = {
    'Booking ID':'Total Bookings','Booking Value': 'Avg Fare Amount',
    'Ride Distance': 'Avg Ride Distance','Driver Ratings':'Avg Driver Ratings',
    'Wait time ration':'Avg wait time'}).round(2)

print("Performance by Vehicle Type:")
print(vehicle_stats.sort_values('Total Bookings', ascending=False))

# 4.Location Analysis
location_stats = df.groupby('Pickup Location').agg({
    'Booking ID': 'count',
    'Customer ID': 'count',
    'Booking Value': 'mean',
    'Wait time ratio':'mean'
}).rename(columns = {'Booking ID':'Total Bookings','Customer ID':'Total Customers',
                    'Booking Value':'Avg Fare Amount','Wait time ratio': 'Avg wait time'}).round(2)

print("Analysis by Pickup Location: Top 10")
print(location_stats.sort_values('Total Bookings', ascending=False).head(10))


# RELATIONSHIP & CORRELATION ANALYSIS
