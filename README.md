# Sentiment-Analysis-News-Headlines
This project aims to perform sentiment analysis on news headlines using machine learning and deep learning algorithms. The steps involved are dataset preparation, model training, hyperparameter tuning, data visualization, and model testing. Multiple models, including Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN), and Deep Learning with Keras, will be implemented and evaluated based on their performance. The goal is to predict whether a news headline expresses positive or negative sentiment.

## Background

With the exponential growth of data being published on various platforms such as comments, blogs, and social media, manually analyzing this vast amount of textual information is challenging. Sentiment analysis provides a solution by automatically and efficiently identifying human emotions or sentiments present in the text. In this project, we focus on sentiment analysis of news headlines to understand public opinion and gain insights from customer feedback.

## Problem Statement

The main objectives of this project are as follows:

1. Distinguish between positive and negative news headlines.
2. Perform sentiment analysis of news articles using different machine learning algorithms.
3. Gain insights from customer feedback and understand public opinion on various issues.

## Project Workflow

The project follows the following workflow:

1. **Dataset Preparation:** Collect and preprocess the dataset of news headlines with corresponding sentiment labels.
2. **Installing Dependencies:** Set up the required libraries and dependencies for running the project.
3. **Training the Model:** Apply machine learning and deep learning algorithms to train the sentiment analysis model on the prepared dataset.
4. **Hyperparameter Tuning:** Optimize the model's performance by tuning hyperparameters.
5. **Visualizing Data:** Visualize the data to gain insights and understand the distribution of sentiments in the dataset.
6. **Testing the Model:** Evaluate the trained model's performance on the test dataset to measure accuracy and F1 score.

##  Model Implementation and Evaluation


## Data Preprocessing

The project begins with data preprocessing steps, which include loading the news headlines dataset from a CSV file and performing cleaning operations such as dropping unnecessary columns and renaming columns. The data is then split into training and testing sets.

## Logistic Regression

The Logistic Regression model utilizes a supervised learning approach. It preprocesses the text data by performing tokenization, lemmatization, removal of stopwords and punctuation, and then represents the text data using TF-IDF vectorization. The model is trained using logistic regression, with hyperparameter tuning using GridSearchCV. Evaluation metrics such as F1 score, train accuracy, and test accuracy are calculated to assess the model's performance.

## Naive Bayes

The Naive Bayes model employs the Complement Naive Bayes algorithm. Hyperparameter tuning is performed using GridSearchCV to find the best values for the hyperparameters. The model is then trained and evaluated using classification reports, F1 score, and accuracy. Predictions can be made on new instances by transforming the new data and using the trained classifier.

## K-Nearest Neighbors (KNN)

The KNN model utilizes the KNeighborsClassifier algorithm. Hyperparameters such as leaf size, number of neighbors, and p-value are tuned using GridSearchCV. The model is trained and evaluated, and predictions can be made on new instances.

## Deep Learning with Keras

The Deep Learning model is implemented using Keras, a high-level neural networks API running on top of TensorFlow. The text data is tokenized and padded to ensure consistent input length. A sequential model is defined, consisting of an embedding layer, global average pooling layer, and dense layers. The model is compiled with binary cross-entropy loss and Adam optimizer. Training is performed for a specified number of epochs, and predictions can be made on new instances. Evaluation metrics include classification reports, accuracy, and loss.

## Evaluation and Comparison

The performance of each model is evaluated using various metrics, including F1 score, accuracy, and loss. The results are presented in classification reports and visualized using bar plots. The comparison reveals that Logistic Regression and Deep Learning with Keras yield the most effective results for sentiment analysis on news headlines, while KNN has the least impact.

## Results

The following results were obtained from the evaluation of different models:

- Logistic Regression:
  - Accuracy: 90.92%
  - F1 Score: 0.92

- Naïve Bayes:
  - Accuracy: 88.86%
  - F1 Score: 0.91

- KNN (K-Nearest Neighbors):
  - Accuracy: 77.10%
  - F1 Score: 0.81

- Deep Learning:
  - Accuracy: 99.4%
  - F1 Score: 0.97

Among the tested models, logistic regression and deep learning showed good performance in classifying positive and negative news headlines, while KNN had the least impact on sentiment analysis.

## Conclusion

This project demonstrates the application of sentiment analysis to investigate human emotions in news headlines. By training various machine learning and deep learning models, we built a binary classifier to distinguish between positive and negative sentiments. The results obtained show promising accuracy and F1 scores, indicating the effectiveness of the models in sentiment classification.