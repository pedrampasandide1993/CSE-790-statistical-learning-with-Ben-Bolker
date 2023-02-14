import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("https://hastie.su.domains/ElemStatLearn/datasets/prostate.data", header=0, delimiter="\t")

# select all columns except for the first column

print(df.shape)
df = df.iloc[:, 1:]

# Split the data into train and test sets
train_df = df[df['train'] == 'T']
test_df = df[df['train'] == 'F']

# Get the features and target variable for the train and test sets
X_train = train_df.iloc[:, 0:-2].values
y_train = train_df.iloc[:, -2].values
X_test = test_df.iloc[:, 0:-2].values
y_test = test_df.iloc[:, -2].values

# Scale the features to have mean zero and variance 96
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Perform data augmentation by adding Gaussian noise to the training data
noise_mean = 0
noise_std = 0.5
num_augmentations = 50
X_train_aug = np.zeros((num_augmentations * len(X_train), X_train.shape[1]))
y_train_aug = np.zeros(num_augmentations * len(y_train))
for i in range(len(X_train)):
    for j in range(num_augmentations):
        X_train_aug[i * num_augmentations + j] = X_train[i] + np.random.normal(noise_mean, noise_std, X_train.shape[1])
        y_train_aug[i * num_augmentations + j] = y_train[i]

# Fit a Ridge regression model on the augmented data
alpha = 10 # regularization parameter
model = Ridge(alpha=alpha)
model.fit(X_train_aug, y_train_aug)
# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE on test set with Ridge Regression by Data augmentation: ", mse)

# Fit a Ridge regression model on the train set
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE on test set with Ridge Regression without Data augmentation: ", mse)

# Fit a linear regression model on the augmented data
model = LinearRegression()
model.fit(X_train_aug, y_train_aug)
# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE on test set with Linear Regression by Data augmentation: ", mse)

# Fit a linear regression model on the augmented data
model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE on test set with Linear Regression without Data augmentation: ", mse)