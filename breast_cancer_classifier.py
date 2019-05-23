from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Explore the data stored in sklearn.datasets
breast_cancer_data = load_breast_cancer()

#Splitting the data into Training and Validation Sets
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
 ## Check to see if our function worked correctly by printing the total number of data and labels.  They should both equal each other. 
print(len(training_data), len(training_labels))

## Run the Classifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(training_data, training_labels)
#find out how accurate our validation set is. 
print(classifier.score(validation_data, validation_labels))

#Use a for loop to change the value of K
accuracies = []
for k in range(1, 101):
	classifier = KNeighborsClassifier(n_neighbors = k)
	classifier.fit(training_data, training_labels)
	accuracies.append(classifier.score(validation_data, validation_labels))
#Graph your results
k_list = range(1,101)

plt.plot(k_list, accuracies)
plt.xlabel('k - values')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
