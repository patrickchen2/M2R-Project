"""
Weather regression models for Dawid-Skene stacking project.
Dataset: london_weather.csv
Target: tomorrow's mean_temp
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("london_weather.csv")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df = df.sort_values("date").reset_index(drop=True)

df["month"] = df["date"].dt.month

df["mean_temp_yesterday"] = df["mean_temp"].shift(1)
df["mean_temp_7d_avg"]    = df["mean_temp"].shift(1).rolling(7).mean()
df["precip_7d_avg"]       = df["precipitation"].shift(1).rolling(7).mean()


df["target"] = df["mean_temp"].shift(-1)

# drop rows where any lag/rolling value is missing (first ~7 rows and last row)
df = df.dropna().reset_index(drop=True)

features = ["cloud_cover", "sunshine", "global_radiation",
            "max_temp", "min_temp", "precipitation",
            "pressure", "snow_depth", "mean_temp", "month",
            "mean_temp_yesterday", "mean_temp_7d_avg", "precip_7d_avg"]

X = df[features].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, shuffle=True
)


#Model 1: Constant - always predicts the training mean

class ConstantModel:
    def fit(self, X, y):
        self.mean = y.mean()
        return self
    def predict(self, X):
        return np.full(len(X), self.mean)


# Model 2: Persistence
# Predicts tomorrow = today's mean_temp.

class PersistenceModel:
    def fit(self, X, y):
        return self
    def predict(self, X):
        return X[:, 8]


# Model 3: Seasonal - average temp for that month

class SeasonalModel:
    def fit(self, X, y):
        self.month_map = {}
        months = X[:, 9].astype(int)
        for m in range(1, 13):
            mask = months == m
            self.month_map[m] = y[mask].mean() if mask.sum() > 0 else y.mean()
        return self
    def predict(self, X):
        months = X[:, 9].astype(int)
        return np.array([self.month_map[m] for m in months])


# Model 5: Sunshine threshold
# Uses only sunshine hours. Sunny: warmer tomorrow. Cloudy: colder.

class SunshineThreshold:
    def fit(self, X, y):
        sun = X[:, 1]
        self.hi = np.percentile(sun, 70)
        self.lo = np.percentile(sun, 30)
        self.sunny_mean  = y[sun >= self.hi].mean()
        self.cloudy_mean = y[sun <= self.lo].mean()
        self.mid_mean    = y.mean()
        return self
    def predict(self, X):
        sun = X[:, 1]
        out = np.full(len(X), self.mid_mean)
        out[sun >= self.hi] = self.sunny_mean
        out[sun <= self.lo] = self.cloudy_mean
        return out


# Model 6: 1-Nearest Neighbour
# No scaling so pressure dominates.

class NearestNeighbour:
    def fit(self, X, y):
        self.Xtr = X.copy()
        self.ytr = y.copy()
        return self
    def predict(self, X):
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            dists = np.sum((self.Xtr - x) ** 2, axis=1)
            out[i] = self.ytr[np.argmin(dists)]
        return out


# Model 7: Naive linear
# Uses only 3 roughly independent features: mean_temp (today), sunshine, pressure.
# Computes a slope for each from its correlation with y, then adds contributions.

class NaiveLinear:
    def fit(self, X, y):
        # columns 8, 1, 6 = mean_temp, sunshine, pressure
        self.cols = [8, 1, 6]
        Xs = X[:, self.cols]
        self.y_mean = y.mean()
        self.x_mean = Xs.mean(axis=0)
        x_std = Xs.std(axis=0) + 1e-6
        y_std = y.std()
        corrs = np.array([np.corrcoef(Xs[:, j], y)[0, 1] for j in range(Xs.shape[1])])
        self.slopes = corrs * y_std / x_std
        return self
    def predict(self, X):
        Xs = X[:, self.cols]
        return self.y_mean + (Xs - self.x_mean).dot(self.slopes)


# Model 8: Decision tree

class DecisionTree:
    def fit(self, X, y):
        self.model = DecisionTreeRegressor(max_depth=3, random_state=67)
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)


# Model 9: Linear regression

class LinearModel:
    def fit(self, X, y):
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = LinearRegression()
        self.model.fit(Xs, y)
        return self
    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


# Train all models

models = [
    ("Constant (training mean)",       ConstantModel()),
    ("Persistence (today = tomorrow)", PersistenceModel()),
    ("Seasonal (monthly average)",     SeasonalModel()),
    ("Sunshine threshold",             SunshineThreshold()),
    ("1-Nearest Neighbour",            NearestNeighbour()),
    ("Naive linear (correlations)",    NaiveLinear()),
    ("Decision tree",                  DecisionTree()),
    ("Linear regression",              LinearModel())
]

for name, model in models:
    model.fit(X_train, y_train)


# MAE = average degrees Celsius you are wrong by.

print(f"  Predicting tomorrow's mean_temp   (MAE in C): \n")
for name, model in models:
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"  {name}  {mae} C")