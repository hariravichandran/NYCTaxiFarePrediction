# import scikit-learn datasets
from sklearn import datasets

# load diabetes dataset with X, y defined
X,y = datasets.load_diabetes(return_X_y=True)

# import XGBoost regressor
from xgboost import XGBRegressor

# import cross_val_score for cross-validation
from sklearn.model_selection import cross_val_score

# score XGBRegressor (ojbective='reg:squarederror' may silence a warning)
scores = cross_val_score(XGBRegressor(objective='reg:squarederror'), X, y, scoring='neg_mean_squared_error')

# get root mean squared error
(-scores)**0.5

print((-scores)**0.5)