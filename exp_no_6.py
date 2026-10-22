
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Read CSV file
msg = pd.read_csv('naivetext.csv')

# Remove extra spaces from column names
msg.columns = msg.columns.str.strip()

# Rename columns to standard names
msg = msg.rename(columns={
    'Text Documents': 'message',
    'Label': 'label'
})

# Remove rows with missing messages or labels
msg = msg.dropna(subset=['message', 'label'])

# Convert messages to strings and clean labels
msg['message'] = msg['message'].astype(str).str.strip()
msg['label'] = msg['label'].astype(str).str.strip().str.lower()

# Remove empty messages
msg = msg[msg['message'] != '']

# Convert labels to numbers
msg['labelnum'] = msg['label'].map({'pos': 1, 'neg': 0})
msg = msg.dropna(subset=['labelnum'])

print('The dimensions of the dataset:', msg.shape)

X = msg['message']
y = msg['labelnum'].astype(int)

print(X)
print(y)

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print('\nThe total number of Training Data:', ytrain.shape)
print('\nThe total number of Test Data:', ytest.shape)

# Convert text to numerical features
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

print('\nThe words or Tokens in the text documents:\n')
print(count_vect.get_feature_names_out())

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(xtrain_dtm, ytrain)

# Predict test results
predicted = clf.predict(xtest_dtm)

# Evaluate classifier
print('\nAccuracy of the classifier is:',
      metrics.accuracy_score(ytest, predicted))

print('\nConfusion matrix:')
print(metrics.confusion_matrix(ytest, predicted, labels=[0, 1]))

print('\nThe value of Precision:',
      metrics.precision_score(ytest, predicted, zero_division=0))

print('\nThe value of Recall:',
      metrics.recall_score(ytest, predicted, zero_division=0))
