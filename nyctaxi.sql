// setup.sql
// there are more columns in AnimeList.csv but i am using a simplified dataset for now
// Data may need to be cleaned further before import
//key,fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count
//2010-05-28 08:18:34.0000004,10.5,2010-05-28 08:18:34 UTC,-73.985881,40.7383,-73.96499,40.775237,1
//123456789012345678901234567
CREATE TABLE fares (
  fareid varchar(30) PRIMARY KEY not null,
  fare_amount numeric,
  pickup_datetime date,
  pickup_longitude date,
  pickup_latitude numeric,
  dropoff_longitude numeric,
  dropoff_latitude numeric,
  passenger_count INT
);

