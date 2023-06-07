# Sentiment-Analysis-News-Headlines
This project aims to perform sentiment analysis on news headlines to determine whether they are positive or negative. The sentiment analysis is carried out using various machine learning and deep learning algorithms to build a binary classifier. The project involves steps such as dataset preparation, model training, hyperparameter tuning, data visualization, and model testing.

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