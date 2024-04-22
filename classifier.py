import preprocess
import neural_network
import constants
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = preprocess.preprocess('codon_usage.csv', test_size=0.2, random_state=42)

classifier = None
prompt = 'Enter a type of classifier to run: rf (random forest), knn (k-nearest neighbors), nn (neural network)\n'
classifier_type = input(prompt)
while classifier == None:
    # A valid classifier object presents the fit(numpy.ndarray, pandas.Series) -> None and predict(pandas.Series) -> numpy.ndarray functions
    if classifier_type.lower() in ['rf', 'random forest']:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type.lower() in ['knn', 'k-nearest neighbors']: # Increasing n_neighbors consistently decreases accuracy and most f1-scores
        classifier = KNeighborsClassifier(n_neighbors=1)
    elif classifier_type.lower() in ['nn', 'neural network']:
        classifier = neural_network.neural_network(input_size=X_train.shape[1], output_size=len(constants.kingdoms), categories=constants.kingdoms)
    else:
        print('That is not a valid classifer type.\n')
        classifier_type = input(prompt)

print('\nTraining...\n')

# Training the chosen classifier
classifier.fit(X_train, y_train)

# Making predictions
y_pred = classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:\n', classification_report(y_test, y_pred, zero_division=0))
