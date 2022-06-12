# access heart disease dataset
url = 'https://media.githubusercontent.com/media/PacktPublishing/Hands-On-Gradient-Boosting-with-XGBoost-and-Scikit-learn/master/Chapter02/heart_disease.csv'

# import pandas
import pandas as pd

# read csv file as dataframe df
df = pd.read_csv(url)

# choose all columns except the last for X
X = df.iloc[:, :-1]

# choose the last column for y
y = df.iloc[:, -1]

# import XGBoost classifier
from xgboost import XGBClassifier

# import cross_val_score for cross-validation
from sklearn.model_selection import cross_val_score

# score XGBClassifier
print(cross_val_score(XGBClassifier(), X, y))
