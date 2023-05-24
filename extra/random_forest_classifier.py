import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('winequality-red.csv')

# Check for missing values
data.isna().sum()

# Remove duplicate data
data.drop_duplicates(keep='first')

# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)

# Drop the target variable
x = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Create an instance of the random forest classifier
rf_model = RandomForestClassifier()

# Fit the model to the training data
rf_model.fit(x_train, y_train)

# Calculate feature importances
importances = rf_model.feature_importances_

# Create a dataframe with feature names and their importance scores
feature_importances = pd.DataFrame({'Feature': x.columns, 'Importance': importances})

# Sort the features by their importance scores in descending order
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Select the top k relevant features
k = 5  # Number of top features to select
selected_features = feature_importances.head(k)['Feature'].tolist()

# Filter the training and testing data to include only the selected features
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

# Create an instance of the logistic regression model
logreg_model = LogisticRegression()

# Fit the model to the training data with selected features
logreg_model.fit(x_train_selected, y_train)

# Predict the labels of the test set with selected features
y_pred = logreg_model.predict(x_test_selected)

# Create confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Compute the precision of the model
precision = precision_score(y_test, y_pred)

# Compute the recall of the model
recall = recall_score(y_test, y_pred)

# Compute the F1 score of the model
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)