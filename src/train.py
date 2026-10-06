from sklearn.datasets import load_iris # Load the Iris Dataset
iris = load_iris()
X = iris.data #shape (150,4)
y = iris.target #shape (150,0)
print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split #Split into Training and Test Sets
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier #Initialise the Model
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train) #Train (Fit) the Model

y_pred = model.predict(X_test) #Making Predictions

#Inspecting Predictions
print("Predictions:",y_pred[:5])
print("True labels:",y_test[:5])

from sklearn.metrics import accuracy_score #Accuracy Metric
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix #Confusion Matrix
cm = confusion_matrix(y_test, y_pred) 
print(cm)

import matplotlib.pyplot as plt #Display confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Iris Classifier Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
import os
os.makedirs("outputs", exist_ok=True)
plt.savefig ("outputs/confusion_matrix.png")

plt.xticks(range(3), iris.target_names)
plt.yticks (range (3), iris.target_names)

for i in range (3):
    for j in range (3):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.show()