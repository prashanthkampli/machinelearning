import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

removed = [0, 50, 100]
new_target = np.delete(iris.target, removed)
new_data = np.delete(iris.data, removed, axis=0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(new_data, new_target)

prediction = clf.predict(iris.data[removed])

print("Original Labels:", iris.target[removed])
print("Labels Predicted:", prediction)

tree.plot_tree(clf)
