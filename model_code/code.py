# Import necessary libraries
import numpy as np   
import pandas as pd  
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Step 3: Ensure the model directory exists
os.makedirs('model', exist_ok=True)  # Creates the directory if it doesn’t exist

# Step 4: Load and Prepare the Data
df = pd.read_csv('diabetes.csv')
df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Separate features (X) and target (Y)
X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']]
Y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Step 5: Define Individual Classifiers for Voting
rf = RandomForestClassifier(n_estimators=200)
lr = LogisticRegression()
svm = SVC(probability=True)

# Step 6: Create and Fit the Voting Classifier
voting_classifier = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='soft')
voting_classifier.fit(X_train, Y_train)

# Step 7: Evaluate the Model's Performance
accuracy = voting_classifier.score(X_test, Y_test)
print("Accuracy of Voting Classifier:", accuracy)

# Step 8: Save the Model with the Latest Version of scikit-learn
filename = 'model/voting_diabetes.pkl'
pickle.dump(voting_classifier, open(filename, 'wb'))
print("Model retrained and saved successfully with scikit-learn 1.5.1.")
