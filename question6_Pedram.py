import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# reading the test set
testset = pd.read_table('zip.test', sep='\s', header=None)
# print(testset)

# reading the train set
trainset = pd.read_table('zip.train', sep='\s', header=None)
# print(trainset.head())


# Keep only 2's and 3's
train = trainset.loc[(trainset.iloc[:, 0] == 2) | (trainset.iloc[:, 0] == 3)]
test = testset.loc[(testset.iloc[:, 0] == 2) | (testset.iloc[:, 0] == 3)]

# Create the train set
x_train = []
for i in range(0, len(train)):
    x_train.append(train.iloc[i, 1:].values.tolist())
y_train = train.iloc[:, 0].values.tolist()

# Create the test set
x_test = []
for i in range(0, len(test)):
    x_test.append(test.iloc[i, 1:].values.tolist())
y_test = test.iloc[:, 0].values.tolist()

# Train the LR model on the train set
LR = LogisticRegression(max_iter=200)
LR.fit(x_train, y_train)
print("Logistic Regression error =", sum(y_train-LR.predict(x_train))/len(y_train))
print("Logistic Regression test error =", sum(y_test-LR.predict(x_test))/len(y_test))
print("Logistic Regression score =", LR.score(x_train, y_train))


# Train the KNN model on the train set for K=1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(x_train, y_train)

print("KNN error when K=1 is =", sum(y_train-knn1.predict(x_train))/len(y_train))
print("KNN test error when K=1 is =", sum(y_test-knn1.predict(x_test))/len(y_test))
print("KNN score when K=1 is =", knn1.score(x_train, y_train))

# Train the KNN model on the train set for K=3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train, y_train)

print("KNN error when K=3 is =", sum(y_train-knn3.predict(x_train))/len(y_train))
print("KNN test error when K=3 is =", sum(y_test-knn3.predict(x_test))/len(y_test))
print("KNN score when K=3 is =", knn3.score(x_train, y_train))

# Train the KNN model on the train set for K=5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(x_train, y_train)

print("KNN error when K=5 is =", sum(y_train-knn5.predict(x_train))/len(y_train))
print("KNN test error when K=5 is =", sum(y_test-knn5.predict(x_test))/len(y_test))
print("KNN score when K=5 is =", knn5.score(x_train, y_train))

# Train the KNN model on the train set for K=7
knn7 = KNeighborsClassifier(n_neighbors=7)
knn7.fit(x_train, y_train)

print("KNN error when K=7 is =", sum(y_train-knn7.predict(x_train))/len(y_train))
print("KNN test error when K=7 is =", sum(y_test-knn7.predict(x_test))/len(y_test))
print("KNN score when K=7 is =", knn7.score(x_train, y_train))


# Train the KNN model on the train set for K=15
knn15 = KNeighborsClassifier(n_neighbors=15)
knn15.fit(x_train, y_train)

print("KNN error when K=15 is =", sum(y_train-knn15.predict(x_train))/len(y_train))
print("KNN test error when K=15 is =", sum(y_test-knn15.predict(x_test))/len(y_test))
print("KNN score when K=15 is =", knn15.score(x_train, y_train))


