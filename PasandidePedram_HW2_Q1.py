import numpy as np
import time
import matplotlib.pyplot as plt


# Define a function to perform linear regression using QR decomposition and others
def linear_regression_qr(X, y):
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    return beta


def linear_regression_svd(X, y):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    beta = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y
    return beta


def linear_regression_naive(X, y):
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def linear_regression_cholesky(X, y):
    L = np.linalg.cholesky(X.T @ X)
    beta = np.linalg.solve(L, X.T @ y)
    beta = np.linalg.solve(L.T, beta)
    return beta


# Define a function to evaluate the mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Generate random data for X and y
np.random.seed(0)
# [1000, 10000, 100000, 1000000, 10000000, 100000000]
sample_sizes = np.array([1000, 10000, 100000, 1000000, 10000000, 100000000])

runtime_QR = []
runtime_SVD = []
runtime_N = []
runtime_Ch = []

mse_QR = []
mse_SVD = []
mse_N = []
mse_Ch = []


j=0
for i in sample_sizes:
    X = np.random.rand(i, 5)
    y = X @ np.array([1, 2, 3, 4, 5]) + np.random.normal(0, 1, i)
    # Split the data into training and test sets
    X_train, X_test = X[:int(0.8 * i), :], X[int(0.8 * i):, :]
    y_train, y_test = y[:int(0.8 * i)], y[int(0.8 * i):]

    # Perform linear regression using QR decomposition
    start_time = time.time()
    beta_QR = linear_regression_qr(X_train, y_train)
    end_time = time.time()
    runtime_QR.append(end_time - start_time)

    # Perform linear regression using SVD
    start_time = time.time()
    beta_SVD = linear_regression_svd(X_train, y_train)
    end_time = time.time()
    runtime_SVD.append(end_time - start_time)

    # Perform linear regression using the naive method
    start_time = time.time()
    beta_N = linear_regression_naive(X_train, y_train)
    end_time = time.time()
    runtime_N.append(end_time - start_time)

    # Perform linear regression using the Cholesky method
    start_time = time.time()
    beta_Ch = linear_regression_cholesky(X_train, y_train)
    end_time = time.time()
    runtime_Ch.append(end_time - start_time)

    # Make predictions on the test set
    y_pred_QR  = X_test @ beta_QR
    y_pred_SVD = X_test @ beta_SVD
    y_pred_N   = X_test @ beta_N
    y_pred_Ch  = X_test @ beta_Ch

    # Calculate the mean squared error
    mse_QR.append(mean_squared_error(y_test, y_pred_QR))
    mse_SVD.append(mean_squared_error(y_test, y_pred_SVD))
    mse_N.append(mean_squared_error(y_test, y_pred_N))
    mse_Ch.append(mean_squared_error(y_test, y_pred_Ch))

    print("Sample size:", i)

    print("Mean Squared Error QR:", mse_QR[j])
    print("Mean Squared Error SVD:", mse_SVD[j])
    print("Mean Squared Error Naive:", mse_N[j])
    print("Mean Squared Error Cholesky:", mse_Ch[j])

    print("Runtime QR:", runtime_QR[j], "seconds")
    print("Runtime SVD:", runtime_SVD[j], "seconds")
    print("Runtime Naive:", runtime_N[j], "seconds")
    print("Runtime Cholesky:", runtime_Ch[j], "seconds")
    j= j+1

    print("\n")

plt.plot(0.8*sample_sizes, runtime_QR, label="QR")
plt.plot(0.8*sample_sizes, runtime_SVD, label="SVD")
plt.plot(0.8*sample_sizes, runtime_N, label="Naive")
plt.plot(0.8*sample_sizes, runtime_Ch, label="Cholesky")

plt.xscale('log')
plt.xlabel("Sample Size (log scale)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime comparison")
plt.legend()

plt.show()





