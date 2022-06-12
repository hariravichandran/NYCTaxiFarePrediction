import pandas as pd
import numpy as np
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# TODO: Remove nrows
# train_raw = pd.read_csv(os.path.join('all', 'train_01p.csv'), nrows=5000)
# train_raw = pd.read_csv(os.path.join('all', 'train_01p.csv'), nrows=500)
train_raw = pd.read_csv(os.path.join('all', 'train_01p.csv'))


train_raw = train_raw.dropna(axis=0)

train_raw['euclidean_distance'] = train_raw.apply(lambda row: math.sqrt((row.pickup_latitude - row.dropoff_latitude) ** 2 + (row.pickup_longitude - row.dropoff_longitude) ** 2), axis=1)

train_raw['pickup_datetime'] = pd.to_datetime(train_raw['pickup_datetime'])

train_raw['pickup_hour'] = train_raw['pickup_datetime'].dt.hour # Find Pickup Hour (0 - 23), AM/PM counted for
train_raw['pickup_minute'] = train_raw['pickup_datetime'].dt.minute # Find Pickup Hour (0 - 23), AM/PM counted for
train_raw['month'] = train_raw['pickup_datetime'].dt.month # Find Month (1 - 12)
train_raw['dayofweek'] = train_raw['pickup_datetime'].dt.weekday # Find Day of Week (1 - 7)
train_raw['dayofyear'] = train_raw['pickup_datetime'].dt.dayofyear # Find Day of Year (1 - 365)

train_raw['average_fares'] = train_raw['fare_amount'].mean()

labels = train_raw['fare_amount']
features = train_raw.drop('fare_amount', 1)

# TODO: select variables
features = features[['passenger_count', 'euclidean_distance', 'pickup_hour', 'pickup_minute',
                 'month', 'dayofweek', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'dayofyear', 'average_fares']]

# Split the data into training and testing sets (80-10-10 train-test-validation split)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, random_state=607551)
test_features, valid_features, test_labels, valid_labels = train_test_split(test_features, test_labels, test_size=0.50, random_state=607551)

# Run Random Forest Classifier Model
rf = RandomForestRegressor(n_estimators=1000, random_state=607551, n_jobs=8)  # Parallel Processing - 8 threads
rf.fit(train_features, train_labels)

# Make predictions
predictions = rf.predict(test_features)

# Errors and Accuracy

print("Mean Absolute Error: ", mean_absolute_error(test_labels, predictions))
print("Mean Absolute Percentage Error: ", mean_absolute_percentage_error(test_labels, predictions))


# errors = abs(predictions - test_labels)  # Errors
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars.')
#
# mape = 100 * (errors / test_labels)
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
