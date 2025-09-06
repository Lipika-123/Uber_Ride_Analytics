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
# 1.Correlation Heatmap

# Select numerical columns for correlation analysis
numerical_cols = ['Booking Value', 'Ride Distance', 'Avg VTAT', 'Avg CTAT', 
                  'Driver Ratings', 'Customer Rating', 'Uber_commission', 
                  'fare_per_km', 'driver_speed_eff', 'Wait time ratio']

corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.show()

# 2. Scatterplot for Key Relationship

# Relationship between Ride Distance and Booking Value
sns.lmplot(data=df, x='Ride Distance', y='Booking Value', hue='Vehicle Type', 
           height=6, aspect=1.5, scatter_kws={'alpha':0.6})
plt.title('Ride Distance vs. Booking Value by Vehicle Type')
plt.show()

# Relationship between Wait Time and Cancellation
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Booking Status', y='Avg CTAT') # Customer Travel Time?
plt.title('Average Wait Time by Booking Status')
plt.ylabel('Wait Time (mins)')
plt.show()

# 3.Rating Analysis

# Distribution of Driver and Customer Ratings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
df['Driver Ratings'].hist(bins=20, ax=ax1)
ax1.set_title('Distribution of Driver Ratings')
ax1.set_xlabel('Rating')

df['Customer Rating'].hist(bins=20, ax=ax2)
ax2.set_title('Distribution of Customer Ratings')
ax2.set_xlabel('Rating')
plt.show()

# Do higher driver ratings lead to higher customer ratings?
sns.jointplot(data=df, x='Driver Ratings', y='Customer Rating', kind='hex', height=8)
plt.suptitle('Driver Ratings vs. Customer Ratings', y=1.02)
plt.show()

# ADVANCED/WHAT-IF ANALYSIS
# 1.Cancellation Impact Analysis

# What is the total revenue lost due to cancellations?
completed_value = df[df['Booking Status'] == 'Completed']['Booking Value'].sum()
cancelled_value_estimate = df[(df['Booking Status'] == 'Cancelled by Driver')|(df['Booking Status'] == 'Cancelled by Customer')]['Booking Value'].sum() # Assuming this is the value if completed
lost_commission = df[(df['Booking Status'] == 'Cancelled by Driver')|(df['Booking Status'] == 'Cancelled by Customer')]['Uber_commission'].sum()

print(f"Estimated Lost Booking Value from Cancellations: ${cancelled_value_estimate:,.2f}")
print(f"Actual Lost Commission from Cancellations: ${lost_commission:,.2f}")
print(f"Potential Revenue Increase if Cancellations Halved: ${(cancelled_value_estimate * 0.5):,.2f}")

# What if we reduce customer wait time (Avg CTAT) by 20%? Would cancellations decrease?
current_avg_ctat_cancelled = df[(df['Booking Status'] == 'Cancelled by Driver')|(df['Booking Status']== 'Cancelled by Customer')]['Avg CTAT'].median()
print(f"Median wait time for cancelled rides: {current_avg_ctat_cancelled:.1f} minutes")

# Analyze the cancellation rate by wait time buckets
df['Wait_Time_Bucket'] = pd.cut(df['Avg CTAT'], bins=[0, 5, 10, 15, 20, 100], 
                                labels=['0-5min', '5-10min', '10-15min', '15-20min', '20+min'])
cancellation_by_wait = df.groupby('Wait_Time_Bucket')['Booking Status'].apply(lambda x: ((x == 'Cancelled by Driver')|(x=='Cancelled by Customer')).mean() * 100).round(1)

print("\nCancellation Rate by Wait Time Bucket:")
print(cancellation_by_wait)

# 2.Driver Efficiency and Route Optimisation

# Which vehicle type is most efficient for drivers?
driver_profitability = df.groupby('Vehicle Type').agg({
    'Uber_commission': 'mean',      # What Uber makes
    'Booking Value': 'mean',        # Total fare
    'driver_speed_eff': 'mean',     # Driver's speed efficiency
    'Ride Distance': 'mean',        # Average trip length
    'Booking ID' : 'count'       
}).round(2).sort_values('Booking Value', ascending=False)

print("Driver Profitability & Efficiency by Vehicle Type:")
print(driver_profitability)

 # Analyze the most profitable routes (Pickup -> Dropoff)
route_profitability = df.groupby(['Pickup Location', 'Drop Location']).agg({
    'Booking ID': 'count',
    'Booking Value': 'mean',
    'Uber_commission': 'mean',
    'Ride Distance': 'mean'
}).round(2).sort_values('Booking Value', ascending=False).head(10)

print("\nTop 10 Most Profitable Routes:")
print(route_profitability)

# 3.Customer Segmentation & Retention

# Identify most valuable customers
customer_value = df.groupby('Customer ID').agg({
    'Booking ID': 'count',
    'Booking Value': 'sum',
    'Cancelled Rides by Customer': 'sum',
    'Customer Rating': 'mean'
}).rename(columns={'Booking ID': 'Total_Bookings', 'Booking Value': 'Total_Spend'}).round(2)

# Segment customers
customer_value['Customer_Segment'] = pd.qcut(customer_value['Total_Spend'], q=3, 
                                             labels=['Low-Value', 'Mid-Value', 'High-Value'])

print("Customer Value Segmentation Summary:")
print(customer_value['Customer_Segment'].value_counts())

# Analyze high-value customer behavior
high_value_customers = customer_value[customer_value['Customer_Segment'] == 'High-Value'].index
high_value_data = df[df['Customer ID'].isin(high_value_customers)]

print("\nPreferred Vehicle Type of High-Value Customers:")
print(high_value_data['Vehicle Type'].value_counts(normalize=True).head(3) * 100)

# 4.Driver Speed Efficiency Analysis
# Analyze how driver speed efficiency correlates with other metrics
plt.figure(figsize=(15, 10))

# Speed efficiency vs Driver Ratings
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='driver_speed_eff', y='Driver Ratings', alpha=0.6)
plt.title('Driver Speed Efficiency vs Ratings')
plt.xlabel('Speed Efficiency Score')
plt.ylabel('Driver Rating')

# Speed efficiency by Vehicle Type
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Vehicle Type', y='driver_speed_eff')
plt.title('Speed Efficiency by Vehicle Type')
plt.xticks(rotation=45)

# Speed efficiency by Hour of Day
plt.subplot(2, 2, 3)
hourly_efficiency = df.groupby('Pickup_hour')['driver_speed_eff'].mean()
hourly_efficiency.plot(kind='bar')
plt.title('Average Speed Efficiency by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Avg Speed Efficiency')

# Correlation heatmap for efficiency metrics
plt.subplot(2, 2, 4)
efficiency_corr = df[['driver_speed_eff', 'Avg VTAT', 'Avg CTAT', 'Wait time ratio', 
                      'Ride Distance', 'Driver Ratings']].corr()
sns.heatmap(efficiency_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Efficiency Metrics Correlation')

plt.tight_layout()
plt.show()

# Identify top and bottom performing drivers by efficiency
driver_efficiency = df.groupby('Driver ID').agg({
    'driver_speed_eff': 'mean',
    'Driver Ratings': 'mean',
    'Booking ID': 'count',
    'Booking Value': 'sum'
}).round(2).sort_values('driver_speed_eff', ascending=False)

print("Top 5 Most Efficient Drivers:")
print(driver_efficiency.head())
print("\nBottom 5 Least Efficient Drivers:")
print(driver_efficiency.tail())

# 5.Wait time Optimisation Analysis
# Deep dive into wait time patterns
df['Wait_Time_Category'] = pd.cut(df['Avg CTAT'], 
                                 bins=[0, 3, 5, 8, 12, 20, 100],
                                 labels=['0-3min', '3-5min', '5-8min', '8-12min', '12-20min', '20+min'])

# Wait time impact on cancellation
wait_time_cancel_analysis = df.groupby('Wait_Time_Category').agg({
    'Booking ID': 'count',
    'Cancelled Rides by Customer': 'sum',
    'Customer Rating': 'mean'
}).assign(
    cancellation_rate=lambda x: (x['Cancelled Rides by Customer'] / x['Booking ID']) * 100
).round(2)

print("Wait Time Impact Analysis:")
print(wait_time_cancel_analysis)

# Geographic analysis of wait times
location_wait_times = df.groupby('Pickup Location').agg({
    'Avg CTAT': 'mean',
    'Booking ID': 'count'
}).sort_values('Avg CTAT', ascending=False).head(10)

print("\nTop 10 Locations with Longest Median Wait Times:")
print(location_wait_times)

# 6. Customer Journey Analysis
# Analyze the complete customer experience funnel
funnel_data = {
    'Stage': ['Total Bookings', 'Completed Rides', 'Cancelled by Customer', 
              'Cancelled by Driver', 'Incomplete Rides'],
    'Count': [
        len(df),
        len(df[df['Booking Status'] == 'Completed']),
        df['Cancelled Rides by Customer'].sum(),
        df['Cancelled Rides by Driver'].sum(),
        df['Incomplete Rides'].sum()
    ]
}

funnel_df = pd.DataFrame(funnel_data)
funnel_df['Percentage'] = (funnel_df['Count'] / funnel_df['Count'].iloc[0]) * 100

print("Customer Journey Funnel Analysis:")
print(funnel_df.round(2))

# Customer lifetime value analysis
customer_behavior = df.groupby('Customer ID').agg({
    'Booking ID': 'count',
    'Booking Value': 'sum',
    'Cancelled Rides by Customer': 'sum',
    'Customer Rating': 'mean',
    'Payment Method': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
}).rename(columns={
    'Booking ID': 'Total_Bookings',
    'Booking Value': 'Lifetime_Value'
}).assign(
    cancel_rate=lambda x: (x['Cancelled Rides by Customer'] / x['Total_Bookings']) * 100
).round(2)

# Segment customers by loyalty and value
customer_behavior['Value_Segment'] = pd.qcut(customer_behavior['Lifetime_Value'], 4, 
                                            labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
customer_behavior['Loyalty_Segment'] = pd.cut(customer_behavior['Total_Bookings'], 
                                              bins = [1,3,5,10],
                                              labels=['New', 'Occasional', 'Regular', 'Frequent'])

print("\nCustomer Segmentation Summary:")
print(pd.crosstab(customer_behavior['Value_Segment'], customer_behavior['Loyalty_Segment']))



