import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
# For training different models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
# For the pipeline
from sklearn.pipeline import Pipeline
# For hyperparameter tuning
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
# To set directory
import os

# Set the working directory to a location of your preference
os.chdir('/Users/mayurpatel/Desktop/MMAI 5400/Assignment 2')

# Read the reviews CSV file
reviews = pd.read_csv('reviews.csv', delimiter='\t')

# Balance the data for positive and negative review
# create the sentiment column and map rating value to sentiment using the dictionary we create below
sentiment_dic = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
reviews['Sentiment'] = reviews['RatingValue'].map(sentiment_dic)

# shorten the positive sentiments to be more aligned with the negative and neutral
pos_temp = reviews[reviews.Sentiment == 2]
docs_to_test = pos_temp[300:305]
pos_temp = pos_temp[:250]


# Get the reviews dataframe but excluding all positive sentiments
reviews_cleaned = reviews[reviews.Sentiment != 2]
# Combine the cleaned dataset
frames = [pos_temp, reviews_cleaned]
# This will be our main dataframe going forward
data = pd.concat(frames)


data = data.iloc[:, -2:]
x = data['Review']
y = data['Sentiment']
# Split the data into training and validation
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Check to see sizes of data with code below to make sure the training and validation are of equal size
# print(x_train.size)
# print(x_val.size)
# print(y_train.size)
# print(y_val.size)

# Convert X and Y data sets into a dataframe and export as csv
x_train_df = x_train.to_frame().reset_index(drop=True)
y_train_df = y_train.to_frame().reset_index(drop=True)
frames_train = [y_train_df, x_train_df]
train = pd.concat(frames_train, axis=1)

x_val_df = x_val.to_frame().reset_index(drop=True)
y_val_df = y_val.to_frame().reset_index(drop=True)
frames_val = [y_val_df, x_val_df]
val = pd.concat(frames_val, axis=1)


# Print the train and validation dataframes to a csv
train.to_csv('train.csv', index=False)
val.to_csv('valid.csv', index=False)

# Read the train and validaion dataframes
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')

# Split datafame's into x and y variables
x_train = train['Review']
y_train = train['Sentiment']
x_val = valid['Review']
y_val = valid['Sentiment']


# First need to turn the text content into numerical feature vectors
# We will do so by using BoW (Bag of words)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)


# The code below will convert the occurances into term frequency times inverse document frequency (TF-IDF)
tfidf_transformer = TfidfTransformer()
# This code will transform the count matrix into a tf-idf representation
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# Fit the model on the modified training data and the ground truth using the y values of training data
nb_model = MultinomialNB().fit(X_train_tfidf, y_train)

# Get some data to practice and test on before the training set
doc_new = docs_to_test['Review']
ground_truth = docs_to_test['Sentiment']

doc_new
docs_new = []
for i in doc_new:
    docs_new.append(i)

docs_new

# Predicting outcome on a new document
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = nb_model.predict(X_new_tfidf)

classificaitons = ['Negative', 'Neutral', 'Positive']

# Building a pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb_model', MultinomialNB()),
])

# Train the model with a single command
text_clf.fit(x_train, y_train)

# Testing our data with MB naive bayes

predicted = text_clf.predict(x_val)
np.mean(predicted == y_val)

# Testing our data with Random forests
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('rf', RandomForestClassifier(random_state=42))
])


text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_val)
np.mean(predicted == y_val)

# HPT with RF
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': [True, False],
    'rf__n_estimators': [100, 200]
}
# Define the grid search
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

# Train the grid search
gs_clf = gs_clf.fit(x_train, y_train)

# Print the best parameters and score
for param_name in sorted(parameters.keys()):
    "%s: %r" % (param_name, gs_clf.best_params_[param_name])


# Testing our data with MLP
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mlp', MLPClassifier()),
])

text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_val)
np.mean(predicted == y_val)


# Testing our data with SVM
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_val)
np.mean(predicted == y_val)


parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(x_train, y_train)

# Hyperparameter tuning with SVM
gs_clf.fit(x_train, y_train)

predicted = gs_clf.predict(x_val)
np.mean(predicted == y_val)
report = metrics.classification_report(y_val, predicted,
                                       target_names=classificaitons, output_dict=True)
precision, recall, f1, support = precision_recall_fscore_support(
    y_val, predicted)
# calculate macro-average F1 score for each class
macro_f1 = dict(zip(classificaitons, f1))


cm = metrics.confusion_matrix(y_val, predicted, normalize='true')


acc = round(accuracy_score(y_val, predicted), 2)
print("Accuracy on the test set: ", acc, "\n")
f1_macro_avg = round(f1_score(y_val, predicted, average='macro'), 2)
print("F1-score on the test set: ", f1_macro_avg, "\n")

print("Class-wise F1 scores:\n negative: ", round(macro_f1['Negative'], 2), "\n neutral: ", round(
    macro_f1['Neutral'], 2), "\n positive: ", round(macro_f1['Positive'], 2), "\n")

# create normalized confusion matrix with labels
print('Normalized confusion matrix:')
confusion_matrix_labelled = pd.DataFrame(
    cm, index=classificaitons, columns=classificaitons)
print(confusion_matrix_labelled)
