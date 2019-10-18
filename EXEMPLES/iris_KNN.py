import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Using pandas to import the dataset
df = pd.read_csv("iris.csv")

# Learn more on pandas read_csv :
#     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# pandas input in general :
#     https://pandas.pydata.org/pandas-docs/stable/reference/io.html


# Spliting dataset between features (X) and label (y)
X = df.drop(columns=["variety"])
y = df["variety"]

# pandas dataframe operations :
#     https://pandas.pydata.org/pandas-docs/stable/reference/frame.html


# Spliting dataset into training set and test set
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# scikit-learn train_test_split :
#     https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Other model selection functions :
#     https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection


# Set algorithm to use
neigh = KNeighborsClassifier(n_neighbors=5)


neigh.fit(X_train, y_train)

# Compute and display the accuracy
accuracy = accuracy_score(y_test, neigh.predict(X_test))

print(accuracy)

# scikit-learn accuracy_score :
#     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# Other scikit-learn metrics :
#     https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
confus_matrice = confusion_matrix(y_test, neigh.predict(X_test))
print(confus_matrice)