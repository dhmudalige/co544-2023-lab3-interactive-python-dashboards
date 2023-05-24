import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('winequality-red.csv')

# Remove duplicate data
data.drop_duplicates(keep='first', inplace=True)

# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)

# Separate the features and labels
x = data.drop('quality', axis=1)
y = data['quality']

# Create an instance of the logistic regression model
logreg_model = LogisticRegression()

# Fit the model to the data
logreg_model.fit(x, y)

# Get the coefficients (weights) of each feature
coefficients = logreg_model.coef_[0]

# Print the coefficients for each feature
for feature, coef in zip(x.columns, coefficients):
    print(f"{feature}: {coef}")
