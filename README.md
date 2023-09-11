# Sentiment-Analysis
Identifying the sentiment of customer reviews based on restaurant dishes

This task was done by training and testing a BoW (Bag-of-words) text classifier on the review texts using RatingValue as labels. 
The ratings are binned into negative (ratings 1 & 2 ), neutral (rating 3 ) and positive (ratings 4 & 5 ) sentiment. 
The binned ratings are coded with negative as 0 , neutral as 1 and positive as 2. 
This new column is called Sentiment.

# Methodology
The ratings are very unbalanced, there are many more positive ( 2 ) ratings than negative ( 0 ) ratings. We drop some
positive ratings to balance the data so we have an approximately equal numbers of negative, neutral and positive
ratings.

The data set is split into training and validation sets and save them as train.csv and valid.csv. 
Thevalidation data is used for model selection and evaluation.

# Evaluation
 Performance is reportied using accuracy, F1-score and a confusion matrix.
