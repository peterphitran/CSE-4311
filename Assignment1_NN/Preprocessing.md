# Taxi Data Preprocessing

The taxi model uses six input features:
- vendor_id 
- passenger_count
- pickup longitude
- pickup latitude
- dropoff longitude
- dropoff latitude

Went with these as when analyzed were most influential when predicting a taxi trip's duration

Training and validation data is set with an 80/20 split and seed=42

Each input feature is standardized with a z-score so inputs are able to be fairly comparable and target values are log transformed (log1p) to make sure outliers dont overtake or skew the model

Originally had tried inputing data raw without using z-score normalization and this caused R^2 to consistently sit around 0 and the model loss would keep stop near the same point of 0.635 which signals to a lack of relationship learned within training. Then tried not transforming target values thus using trip duration in seconds, however this caused the long trips to have immense impact since values get squared within the model.

program makes training-validation loss plots for 8, 16, and 32 hidden nodes, each tested with ReLU and Tanh.

