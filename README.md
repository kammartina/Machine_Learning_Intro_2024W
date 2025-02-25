# Machine_Learning_Intro_2024W
This repository contains code that is part of programming assignments in **Introduction to Machine Learning for Language Processing in 2024W**.
<br><br>

***Exercise 1: Python Basics***
<br>
This notebook is an introduction to Natural Language Processing (NLP) using Python and the NLTK library. In summary, this introductory machine learning class notebook focuses on **using Python and NLTK for basic NLP tasks such as text exploration, frequency analysis, and n-gram identification**. It lays the foundation for further exploration of more advanced NLP techniques.
<br><br>
Goals:
<br>
Introduce the NLTK library and its functionality: The notebook demonstrates how to use NLTK for basic text analysis tasks.
Understand text analysis basics: It covers fundamental concepts like tokenization, frequency distribution, and vocabulary analysis using the "Moby Dick" text as an example.
Learn about n-grams and collocations: It explains the concepts of n-grams and collocations, including how to identify them using NLTK functions.
<br><br>
What has been done:
- Setup: The notebook starts by importing necessary libraries (NLTK) and downloading required resources.
- Exploratory Data Analysis (EDA): It explores the "Moby Dick" dataset by looking at the first and last sentences, calculating vocabulary size, and analyzing word frequency distribution.
- Filtering: The notebook demonstrates how to filter out stop words and punctuation to focus on more meaningful words.
- Word Length Analysis: It analyzes the frequency distribution of word lengths within the text.
- N-grams and Collocations: The notebook shows how to identify bigrams and collocations in the text to understand word relationships.
- Python and NLP: It highlights Python's built-in string processing capabilities for NLP tasks like searching for substrings within tokens.
<br><br><br>

***Exercise 2: Gender Identification With Decision Tree***
<br>
The notebook guides through the process of **building a simple machine learning model that can predict the gender of a name - male/female**. This involves **data exploration, feature engineering, model training, and evaluation**. The notebook demonstrates how to apply basic machine learning techniques to a text-based classification problem. It highlights the importance of feature engineering and data analysis in improving model performance. The final exercise encourages further exploration and experimentation to achieve better results.
<br><br>
What has been done:
<br>
- Data Exploration and Feature Engineering: The initial goal is to explore this data, identify patterns, and extract meaningful features that can be used to distinguish between male and female names. These features are things like the last letter of the name, the length of the name, or the presence of specific letter combinations.
- Building a Classifier with NLTK: Building a basic classifier using the NLTK library. Creating a feature extraction function (gender_features) that takes a name and returns a dictionary of features. Training a decision tree classifier on the extracted features. Evaluating the classifier's accuracy.
- Building a Classifier with Scikit-learn: The notebook then introduces Scikit-learn, a more comprehensive machine learning library. The classification task is repeated using Scikit-learn's decision tree classifier. Converting the features into a numerical format suitable for Scikit-learn. Using train_test_split to divide the data into training and testing sets. Training the classifier and evaluating its performance using metrics like precision, recall, and F1-score.
- Improving the Classifier: The final exercise challenges to improve the classifier's performance by:
  - Performing a deeper statistical analysis of the names, looking for patterns in letter combinations (n-grams).
  - Creating new features based on your analysis and retraining the model.
  - Aiming for higher accuracy scores.
 <br><br><br>
 
***Exercise 3: Spam Detection Using Naive Bayes Classification***
<br>
This project explores probabilistic modeling for Natural Language Processing (NLP) by **building a spam detection system**. It demonstrates the **application of Naive Bayes classifiers**, specifically **MultinomialNB and ComplementNB**, for classifying emails as spam or not spam. The primary goal of this project is to explore **probabilistic modeling** in NLP, apply it to a real-world problem, and demonstrate the benefits of **hyperparameter tuning and model comparison**. By exploring different models and metrics, the notebook helps understand the **trade-offs in performance and robustness** when dealing with text data.
<br><br>
What has been done:
<be>
- Naive Bayes: A probabilistic machine learning algorithm based on Bayes' theorem.
- Text Preprocessing: Using techniques like CountVectorizer and TF-IDF to convert text into numerical features.
- Model Evaluation: Assessing classifier performance using metrics like precision, recall, F1-score, and Matthews Correlation Coefficient.
- Hyperparameter Tuning: Optimizing the model's performance using Grid Search to find the best smoothing parameter (alpha).
- Model Comparison: Evaluating and comparing the performance of Naive Bayes classifiers against other models like Decision Tree, Random Forest, and k-Nearest Neighbors.
<br><br><be>

***Exercise 4:***
