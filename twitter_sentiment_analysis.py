# -*- coding: utf-8 -*-
"""Twitter-Sentiment-Analysis.ipynb

# Twitter-Sentiment-**Analysis**

Importing necessary libraries for data processing, visualization, and NLP
"""

import numpy as np
import pandas as pd
import nltk

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

train = pd.read_csv('/content/train_tweet.csv')
test = pd.read_csv('/content/test_tweets.csv')

print(train.shape)
print(test.shape)

train.head()

test.head()

train.isnull().any()
test.isnull().any()

# checking out the negative comments from the train set

train[train['label'] == 0].head(10)

# checking out the postive comments from the train set

train[train['label'] == 1].head(10)

"""Plotting the distribution of sentiment labels using a bar chart

"""

train['label'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))

# checking the distribution of tweets in the data

length_train = train['tweet'].str.len().plot.hist(color = 'pink', figsize = (6, 4))
length_test = test['tweet'].str.len().plot.hist(color = 'orange', figsize = (6, 4))

# adding a column to represent the length of the tweet

train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

train.head(10)

"""Displaying summary statistics for each sentiment label

"""

train.groupby('label').describe()

"""Plotting a histogram to show the variation of sentiment labels with respect to text length

"""

train.groupby('len')['label'].mean().plot.hist(color = 'black', figsize = (6, 4),) # Change is here
plt.title('variation of length')
plt.xlabel('Length')
plt.show()

"""Extracting features from text using CountVectorizer while removing stop words

Calculating word frequencies and sorting them in descending order

Creating a DataFrame of word frequencies and plotting the top 30 most frequent words

"""

from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(train.tweet)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Most Frequently Occuring Words - Top 30")

""" Generating a word cloud to visualize the most frequent words in the dataset

"""

from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize = 22)

"""Creating a word cloud for neutral tweets (label = 0) to visualize common words

"""

normal_words =' '.join([text for text in train['tweet'][train['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Neutral Words')
plt.show()

"""Creating a word cloud for negative tweets (label = 1) to visualize common words

"""

negative_words =' '.join([text for text in train['tweet'][train['label'] == 1]])

wordcloud = WordCloud(background_color = 'cyan', width=800, height=500, random_state = 0, max_font_size = 110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()

"""Function to extract hashtags from text using regular expressions

"""

# collecting the hashtags
import re # Importing the 're' module

def hashtag_extract(x):
    hashtags = []

    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

"""Extracting hashtags from neutral (label = 0) and negative (label = 1) tweets

 Flattening the nested lists of extracted hashtags

"""

# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

"""Calculating frequency distribution of hashtags in neutral tweets

Creating a DataFrame and selecting the top 20 most frequent hashtags

Plotting a bar chart to visualize the most common hashtags

"""

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

"""Calculating frequency distribution of hashtags in negative tweets

Creating a DataFrame and selecting the top 20 most frequent hashtags

Plotting a bar chart to visualize the most common negative hashtags

"""

a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

"""Tokenizing tweets into words for further processing

Creating a Word2Vec model to generate word embeddings

Training the model with a skip-gram approach for better context learning

"""

# tokenizing the words present in the training set
tokenized_tweet = train['tweet'].apply(lambda x: x.split())

# importing gensim
import gensim

# creating a word to vector model
# Replacing 'size' with 'vector_size'
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            vector_size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(train['tweet']), epochs=20)

"""Retrieving words most similar to "dinner" based on the trained Word2Vec model

"""

model_w2v.wv.most_similar(positive = "dinner")

model_w2v.wv.most_similar(positive = "cancer")

model_w2v.wv.most_similar(positive = "apple")

model_w2v.wv.most_similar(negative = "hate")

"""Importing tqdm for progress tracking


"""

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import TaggedDocument

# Instead of:
# labeled_sentence = LabeledSentence(words=tokens, tags=[tag])

# Use:
# tagged_document = TaggedDocument(words=tokens, tags=[tag])

"""Function to label each tweet for Doc2Vec training

Using TaggedDocument to associate each tweet with a unique identifier

Labeling all tweets and displaying the first 6 labeled examples

"""

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
# Instead of importing LabeledSentence, import TaggedDocument
from gensim.models.doc2vec import TaggedDocument

def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        # Replace LabeledSentence with TaggedDocument
        output.append(TaggedDocument(s, ["tweet_" + str(i)]))
    return output

# label all the tweets
labeled_tweets = add_label(tokenized_tweet)

labeled_tweets[:6]

"""Importing necessary libraries for text preprocessing

Downloading stopwords for removing common words that add little meaning

Importing PorterStemmer for stemming words to their root form

"""

# removing unwanted patterns from the data

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

"""Initializing an empty list to store preprocessed tweets

Removing non-alphabetic characters from each tweet

Converting text to lowercase for uniformity

Splitting the tweet into individual words

Applying stemming to reduce words to their root form

Removing stopwords to retain only meaningful words

Joining the processed words back into a cleaned sentence

Appending the cleaned tweet to the corpus

"""

train_corpus = []

for i in range(0, 31962):
  review = re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
  review = review.lower()
  review = review.split()

  ps = PorterStemmer()

  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  # joining them back with space
  review = ' '.join(review)
  train_corpus.append(review)

"""Initializing an empty list to store preprocessed test tweets

Removing non-alphabetic characters from each test tweet

Converting text to lowercase for uniformity

Splitting the tweet into individual words

Applying stemming to reduce words to their root form

Removing stopwords to retain only meaningful words

Joining the processed words back into a cleaned sentence

Appending the cleaned tweet to the test corpus

"""

test_corpus = []

for i in range(0, 17197):
  review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
  review = review.lower()
  review = review.split()

  ps = PorterStemmer()

  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  # joining them back with space
  review = ' '.join(review)
  test_corpus.append(review)

"""Importing CountVectorizer to convert text data into a bag-of-words model

Initializing CountVectorizer with a maximum of 2500 features

Transforming the preprocessed training corpus into a numerical feature matrix

Extracting the target variable (labels) from the dataset

Printing the shape of the feature matrix and the target variable for verification

"""

# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]

print(x.shape)
print(y.shape)

"""Importing CountVectorizer to convert text data into a bag-of-words model

Initializing CountVectorizer with a maximum of 2500 features

Transforming the preprocessed test corpus into a numerical feature matrix

Printing the shape of the transformed test dataset for verification

"""

# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)
x_test = cv.fit_transform(test_corpus).toarray()

print(x_test.shape)

"""Importing train_test_split to split the dataset into training and validation sets

Splitting the dataset with 75% for training and 25% for validation

Setting a random state for reproducibility

Printing the shapes of the resulting train and validation sets for verification

"""

# splitting the training data into train and valid sets

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25, random_state = 42)

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)

"""Importing StandardScaler for feature scaling

Fitting and transforming the training data to standardize features

Transforming the validation and test data using the same scaler to maintain consistency

"""

# standardization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

"""Importing RandomForestClassifier for classification
Importing confusion_matrix and f1_score for model evaluation

Initializing and training the RandomForestClassifier model
Predicting the labels for the validation set

Printing training and validation accuracy
Calculating and printing the F1 score to measure model performance

Generating and printing the confusion matrix to evaluate prediction results

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("F1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)

"""Importing LogisticRegression for classification

Initializing and training the Logistic Regression model

Predicting the labels for the validation set

Printing training and validation accuracy

Calculating and printing the F1 score to measure model performance

Generating and printing the confusion matrix to evaluate prediction results

"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)

"""Importing DecisionTreeClassifier for classification

Initializing and training the Decision Tree model

Predicting the labels for the validation set

Printing training and validation accuracy

Calculating and printing the F1 score to measure model performance

Generating and printing the confusion matrix to evaluate prediction results

"""

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)

"""Importing Support Vector Classifier (SVC) from sklearn

Initializing and training the SVM model

Predicting labels for the validation set

Printing training and validation accuracy to evaluate model performance

Calculating and displaying the F1 score for classification effectiveness

Generating and printing the confusion matrix to analyze prediction results

"""

from sklearn.svm import SVC

model = SVC()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)

"""Importing XGBoost Classifier

Initializing and training the XGBoost model

Predicting labels for the validation set

Printing training and validation accuracy to assess model performance

Calculating and displaying the F1 score for model evaluation

Generating and printing the confusion matrix to analyze classification results

"""

from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)
