from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# load the iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.tree import DecisionTreeClassifier

# fit a decision tree model to the training data
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(y_pred)
print(y_test)

confusion_metric_output = confusion_matrix(y_test, y_pred)
print(confusion_metric_output)

display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix = confusion_metric_output, display_labels=iris.target_names)
display_confusion_matrix.plot(cmap="magma")
plt.show()