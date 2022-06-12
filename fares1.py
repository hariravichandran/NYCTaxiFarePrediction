import pandas as pd
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# TODO: Remove nrows
train_raw = pd.read_csv(os.path.join('all', 'train_01p.csv'), nrows=100)

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


# valid = pd.read_csv(os.path.join("all", "test.csv")) #Validation Data is Testing Data from Kaggle
#
# valid['euclidean_distance'] = valid.apply(lambda row: math.sqrt((row.pickup_latitude - row.dropoff_latitude) ** 2 + (row.pickup_longitude - row.dropoff_longitude) ** 2), axis=1)
#
# # Find Euclidean Distance based on longitude - latitude
# # valid['euclidean_distance'] = math.sqrt((valid['pickup_latitude'] - valid['dropoff_latitude']) ^ 2 + (valid['pickup_longitude'] - valid['dropoff_longitude'])^2)
#
# valid['pickup_datetime'] = pd.to_datetime(valid['pickup_datetime'])
#
# valid['pickup_hour'] = valid['pickup_datetime'].dt.hour # Find Pickup Hour (0 - 23), AM/PM counted for
# valid['pickup_minute'] = valid['pickup_datetime'].dt.minute # Find Pickup Hour (0 - 23), AM/PM counted for
# valid['month'] = valid['pickup_datetime'].dt.month # Find Month (1 - 12)
# valid['dayofweek'] = valid['pickup_datetime'].dt.weekday # Find Day of Week (1 - 7)
# valid['dayofyear'] = valid['pickup_datetime'].dt.dayofyear # Find Day of Year (1 - 365)
#
# valid['average_fares'] = valid['fare_amount'].mean()

# Run Random Forest Classifier Model
rf_clf = RandomForestClassifier(criterion='entropy')
rf_clf.fit(train_features, train_labels)




# # Run Random Forest Model - Ranger
# #gbm_control = trainControl(method = "boot", number = 10, allowParallel = TRUE) #Do Boosting Only Three Times
# #gbm_grid = expand.grid(n.trees = 200, interaction.depth = 8, shrinkage = 0.1, n.minobsinnode = 5) #Only 30 iterations as opposed to 150 iterations (default)
#
# rf_control = trainControl(method = "cv", number = 10) #Do Resampling Only Three Times
# rf_grid = expand.grid(mtry = c(1:3), splitrule = "variance", min.node.size = 5) #Only 'mtry' variables for random forest, keep other variables default
#
# f_rf = train(fare_amount ~., data = train_select, method = 'ranger', trControl = rf_control, tuneGrid = rf_grid)
#
# summary(f_rf)
#
# print("Training Results: ")
# p_rf_train = predict(f_rf, train_select)
# print(postResample(pred = p_rf_train, obs = train_select['fare_amount']))
#
# print("Testing Results: ")
# p_rf_test = predict(f_rf, test)
# print(postResample(pred = p_rf_test, obs = test_select['fare_amount']))
#
# #Validation Set - Kaggle Results
# p_rf_valid = predict(f_rf, valid_select)
# result = as.data.frame(cbind(valid['key'], p_rf_valid))
# colnames(result) = c("key", "fare_amount")
# write.csv(result, "fares14_python.csv", row.names = FALSE)





# in_train = createDataPartition(train_raw['fare_amount'], p = 0.8, list = FALSE) # Set aside 20% of our training set for testing

#
# train_raw['pickup_hour'] = hour(train_raw['pickup_datetime']) # Find Pickup Hour (0 - 23), AM/PM counted for
# train_raw['pickup_minute'] = minute(train_raw['pickup_datetime']) # Find Pickup Minute (0 - 59)
# train_raw['month'] = month(train_raw['pickup_datetime']) # Find Month (1 - 12)
# train_raw['dayofweek'] = wday(train_raw['pickup_datetime']) # Find Day of Week (1 - 7)
# train_raw['dayofyear'] = yday(train_raw['pickup_datetime']) # Find Day of Year (1 - 365)
#
# average_fare = mean(train_raw['fare_amount'])
# train_raw['average_fares'] = rep(average_fare, dim(train_raw)[1])
#
# in_train = createDataPartition(train_raw['fare_amount'], p = 0.8, list = FALSE) # Set aside 20% of our training set for testing
#
# train = train_raw[in_train,]
# test = train_raw[-in_train, ]
#
# #rm(in_train, train_raw)
#
# #Select the parameters we want for the model
# train_select = select(train, fare_amount, passenger_count, euclidean_distance, pickup_hour, pickup_minute, month, dayofweek, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, dayofyear, average_fares)
# test_select = select(test, fare_amount, passenger_count, euclidean_distance, pickup_hour, pickup_minute, month, dayofweek, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, dayofyear, average_fares)
#
# valid = fread("all/test.csv") #Validation Data is Testing Data from Kaggle
#
# # Find Euclidean Distance based on longitude - latitude
# valid['euclidean_distance'] = sqrt((valid['pickup_latitude'] - valid['dropoff_latitude']) ^ 2 + (valid['pickup_longitude'] - valid['dropoff_longitude'])^2)
#
# valid['pickup_hour'] = hour(valid['pickup_datetime']) # Find Pickup Hour (0 - 23), AM/PM counted for
# valid['pickup_minute'] = minute(valid['pickup_datetime']) # Find Pickup Minute (0 - 59)
# valid['month'] = month(valid['pickup_datetime']) # Find Month (1 - 12)
# valid['dayofweek'] = wday(valid['pickup_datetime']) # Find Day of Week (1 - 7)
# valid['dayofyear'] = yday(valid['pickup_datetime']) # Find Day of Year (1 - 365)
#
# valid['average_fares'] = rep(average_fare, dim(valid)[1])
#
# #Select the parameters we want for the model
# valid_select = select(valid, passenger_count, euclidean_distance, pickup_hour, pickup_minute, month, dayofweek, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, dayofyear, average_fares)
#
# # Run Random Forest Model - Ranger
# #gbm_control = trainControl(method = "boot", number = 10, allowParallel = TRUE) #Do Boosting Only Three Times
# #gbm_grid = expand.grid(n.trees = 200, interaction.depth = 8, shrinkage = 0.1, n.minobsinnode = 5) #Only 30 iterations as opposed to 150 iterations (default)
#
# rf_control = trainControl(method = "cv", number = 10) #Do Resampling Only Three Times
# rf_grid = expand.grid(mtry = c(1:3), splitrule = "variance", min.node.size = 5) #Only 'mtry' variables for random forest, keep other variables default
#
# f_rf = train(fare_amount ~., data = train_select, method = 'ranger', trControl = rf_control, tuneGrid = rf_grid)
#
# summary(f_rf)
#
# print("Training Results: ")
# p_rf_train = predict(f_rf, train_select)
# print(postResample(pred = p_rf_train, obs = train_select['fare_amount']))
#
# print("Testing Results: ")
# p_rf_test = predict(f_rf, test)
# print(postResample(pred = p_rf_test, obs = test_select['fare_amount']))
#
# #Validation Set - Kaggle Results
# p_rf_valid = predict(f_rf, valid_select)
# result = as.data.frame(cbind(valid['key'], p_rf_valid))
# colnames(result) = c("key", "fare_amount")
# write.csv(result, "fares14_python.csv", row.names = FALSE)