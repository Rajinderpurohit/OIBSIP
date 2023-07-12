# Importing the necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Loading the iris dataset
iris = load_iris()

# Splitting the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
#1=>setosa, 2=>versicolor, 3=>virginica
# Creating the KNN classifier object
knn = KNeighborsClassifier(n_neighbors=1)

# Training the model using the training sets
knn.fit(X_train, y_train)

# Predicting the classes of the test set
y_pred = knn.predict(X_test)

#parameters=> knn.predict([[5.8, 2.8, 5.1, 2.4]])
# Printing the accuracy of the model
print("Accuracy:", knn.score(X_test, y_test))


#This code uses the K-Nearest Neighbors (KNN) algorithm to classify iris flowers based on their measurements.
#The code first loads the iris dataset and then splits it into training and testing sets.
#It then creates a KNN classifier object and trains it using the training sets.
#Finally, it predicts the classes of the test set and prints the accuracy of the model
