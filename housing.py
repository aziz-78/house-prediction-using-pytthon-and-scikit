import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

housing_df = pd.read_csv("housing.csv")
housing_df['total_rooms'] = np.log(housing_df['total_rooms']+1)
housing_df['total_bedrooms'] = np.log(housing_df['total_bedrooms']+1)
housing_df['population'] = np.log(housing_df['population']+1)
housing_df['households'] = np.log(housing_df['households']+1)
housing_df['bedroom_ratio'] = housing_df['total_bedrooms']/housing_df['total_rooms']
housing_df['household_rooms'] = housing_df['total_rooms']/housing_df['households']
housing_df = housing_df.dropna(subset=["total_bedrooms"])
housing_dummies = pd.get_dummies(housing_df["ocean_proximity"])
print(housing_dummies.head())
housing_df = housing_df.drop("ocean_proximity", axis=1)
housing_dummies = pd.concat([housing_df, housing_dummies], axis=1)
X = housing_dummies.drop("median_house_value", axis=1).values
y = housing_dummies["median_house_value"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=21)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred = lin_reg.predict(X_train_scaled)
score = lin_reg.score(X_test_scaled, y_test)
print(score)
