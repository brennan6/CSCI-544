import numpy as np
import pandas as pd
import re
import nltk
import sklearn
import warnings
import sys
from platform import python_version
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

### Python Run Version: 3.9.5
### READ: Include path to dataset as imput to .py file

class run_hw(object):
    """Create this object and run components to print results as requested"""
    def __init__(self, url):
        self.url = url
        self.data = None
        self.sampled_data = None
        self.stop_words = set(stopwords.words('english'))
        self.count_3 = None
        self.train = None
        self.test = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def read_data(self):
        self.data = pd.read_csv(self.url, sep="\t", error_bad_lines=False, warn_bad_lines=False)

    def keep_review_ratings(self):
        self.data = self.data.loc[:, ["review_body", "star_rating"]]

    def labeling_reviews(self):
        # Map star rating to sentiment rating:
        d_ = {4:1, 5:1, 1:0, 2:0, 3:0}
        self.data["sentiment"] = self.data["star_rating"].map(d_)
        
        # Counts of each rating:
        print("Counts of each Rating:")
        print(self.data["star_rating"].value_counts())
        print("\b")

        # Discard ratings of 3:
        self.count_3 = len(self.data[self.data["star_rating"] == 3])
        self.data = self.data[~(self.data["star_rating"] == 3)]

    def select_random_rows(self):
        pos_reviews = self.data[self.data["sentiment"] == 1].sample(100000)
        neg_review = self.data[self.data["sentiment"] == 0].sample(100000)

        print("Counts of each Sentiment Rating:")
        print("Class 0 count:", self.data["sentiment"].value_counts()[0], end=", ")
        print("Class 1 count:", self.data["sentiment"].value_counts()[1], end=", ")
        print("Discarded (3) Class:", self.count_3)
        print("\b")

        self.sampled_data = pd.concat([pos_reviews, neg_review])
        self.sampled_data.reset_index(drop=True, inplace=True)

    def train_test_split(self):
        self.train, self.test = train_test_split(self.sampled_data, test_size=0.2)
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

    @staticmethod
    def contractionfunction(s):
        contraction_d = {"wont":"will not", "dont":"do not", "hasnt":"has not", "havent":"have not",
                        "im":"i am", "id":"i would", "itll": "it will", "ive": "i have", "isnt":"is not",
                        "lets":"let us", "mustve": "must have", "shed":"she would", "shell": "she will",
                        "thats": "that is", "theyd": "they had", "theyll": "they will", "weve": "we have"}
        for k_ in contraction_d.keys():
            if type(s) != str:
                s = str(s)
            s = s.replace(k_, contraction_d[k_])
        return s

    def data_cleaning(self):
        # Find average length of reviews (in characters) before:
        print("Average length of reviews (in characters) before:")
        print("Average Training Before:", np.mean(self.train["review_body"].str.len()), end=", ")
        print("Average Testing Before:", np.mean(self.test["review_body"].str.len()))
        print("\b")

        # Print 3 reviews Before Data Cleaning/Pre-processing:":
        print("3 Reviews Before Data Cleaning/Pre-processing:")
        print(self.train["review_body"][62], end=", ")
        print("Rating:", self.train["star_rating"][62])
        print("\b")
        print(self.train["review_body"][245], end=", ")
        print("Rating:", self.train["star_rating"][245])
        print("\b")
        print(self.train["review_body"][97765], end=", ")
        print("Rating:", self.train["star_rating"][97765])
        print("\b")

        self.train["review_body"] = self.train["review_body"].str.lower()
        self.test["review_body"] = self.test["review_body"].str.lower()

        # remove the HTML and URLs from the reviews:
        self.train["review_body"] = self.train["review_body"].replace(r'http\S+|www.\S+', '', regex=True)
        self.test["review_body"] = self.test["review_body"].replace(r'http\S+|www.\S+', '', regex=True)

        # remove non-alphabetical characters:
        self.train["review_body"] = self.train["review_body"].replace(r'[^a-z|\s]', '', regex=True)
        self.test["review_body"] = self.test["review_body"].replace(r'[^a-z|\s]', '', regex=True)

        # remove the extra spaces between words:
        self.train["review_body"] = self.train["review_body"].replace(r'\s\s+', ' ', regex=True)
        self.test["review_body"] = self.test["review_body"].replace(r'\s\s+', ' ', regex=True)

        self.test["review_body"] = self.test["review_body"].apply(self.contractionfunction)
        self.train["review_body"] = self.train["review_body"].apply(self.contractionfunction)

        # Find average length of reviews (in characters) AFTER Cleaning (but before stopwords/lemmatization):
        print("Average length of reviews (in characters) after data cleaning, before stopwords/lemmatization:")
        print("Average Training Length After Data Cleaning:", np.mean(self.train["review_body"].str.len()), end=", ")
        print("Average Testing Length After Data Cleaning:", np.mean(self.test["review_body"].str.len()))
        print("\b")
    
    @staticmethod
    def remove_stopwords(s):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(s)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return filtered_tokens

    @staticmethod
    def lemmatize(s):
        #Should already be tokenized from above:
        lem_obj = WordNetLemmatizer()
        lemmatized_tokens = [lem_obj.lemmatize(word) for word in s]
        return lemmatized_tokens

    @staticmethod
    def stringify(s):
        # Convert the tokenized values back into strings:
        str_ = " ".join([word for word in s])
        return str_

    def pre_processing(self):
        # remove stopwords:
        self.train["review_body"] = self.train["review_body"].apply(self.remove_stopwords)
        self.test["review_body"] = self.test["review_body"].apply(self.remove_stopwords)

        #perform lemmatization:
        self.train["review_body"] = self.train["review_body"].apply(self.lemmatize)
        self.test["review_body"] = self.test["review_body"].apply(self.lemmatize)

        #return to string formatting:
        self.train["review_body"] = self.train["review_body"].apply(self.stringify)
        self.test["review_body"] = self.test["review_body"].apply(self.stringify)

        # Find average length of reviews (in characters) AFTER stopwords/lemmatize:
        print("Average length of reviews (in characters) after pre-processing:")
        print("Average Training Length After Pre-Processing:", np.mean(self.train["review_body"].str.len()), end=", ")
        print("Average Testing Length After Pre-Processing:", np.mean(self.test["review_body"].str.len()))
        print("\b")

        # Print 3 reviews AFTER stopwords/lemmatize:
        print("3 Reviews After Data Cleaning/Pre-processing:")
        print(self.train["review_body"][62])
        print("\b")
        print(self.train["review_body"][245])
        print("\b")
        print(self.train["review_body"][97765])
        print("\b")

    def tfidf(self):
        reviews_train = self.train["review_body"].values
        reviews_test = self.test["review_body"].values

        vectorizer = TfidfVectorizer()
        self.X_train = vectorizer.fit_transform(reviews_train)
        self.X_test = vectorizer.transform(reviews_test)

        self.Y_train = self.train["sentiment"].values
        self.Y_test = self.test["sentiment"].values

    def perceptron(self):
        print("Perceptron Results:")
        perceptron = Perceptron(tol=1e-3, random_state=42)
        perceptron.fit(self.X_train, self.Y_train)
        print("Training Accuracy:", perceptron.score(self.X_train, self.Y_train), end=", ")

        y_pred = perceptron.predict(self.X_train)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_train, y_pred, average="macro")
        print("Precision (Training):", precision, end=", ")
        print("Recall (Training):", recall, end=", ")
        print("F1 (Training):", f1)

        print("\b")

        print("Test Accuracy:", perceptron.score(self.X_test, self.Y_test), end=", ")
        y_pred_test = perceptron.predict(self.X_test)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_test, y_pred_test, average="macro")
        print("Precision (Test):", precision, end=", ")
        print("Recall (Test):", recall, end=", ")
        print("F1 (Test):", f1)
        print("\b")
        
    def svm(self):
        print("SVM Results:")
        svm = LinearSVC(max_iter=1000)
        svm.fit(self.X_train, self.Y_train)
        print("Training Accuracy:", svm.score(self.X_train, self.Y_train), end=", ")

        y_pred = svm.predict(self.X_train)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_train, y_pred, average="macro")
        print("Precision (Training):", precision, end=", ")
        print("Recall (Training):", recall, end=", ")
        print("F1 (Training):", f1)

        print("\b")

        print("Test Accuracy:", svm.score(self.X_test, self.Y_test), end=", ")
        y_pred_test = svm.predict(self.X_test)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_test, y_pred_test, average="macro")
        print("Precision (Test):", precision, end=", ")
        print("Recall (Test):", recall, end=", ")
        print("F1 (Test):", f1)
        print("\b")

    def logistic_regression(self):
        print("Logistic Regression Results:")
        lr = LogisticRegression()
        lr.fit(self.X_train, self.Y_train)
        print("Training Accuracy:", lr.score(self.X_train, self.Y_train), end=", ")

        y_pred = lr.predict(self.X_train)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_train, y_pred, average="macro")
        print("Precision (Training):", precision, end=", ")
        print("Recall (Training):", recall, end=", ")
        print("F1 (Training):", f1)

        print("\b")

        print("Test Accuracy:", lr.score(self.X_test, self.Y_test), end=", ")
        y_pred_test = lr.predict(self.X_test)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_test, y_pred_test, average="macro")
        print("Precision (Test):", precision, end=", ")
        print("Recall (Test):", recall, end=", ")
        print("F1 (Test):", f1)
        print("\b")

    def naive_bayes(self):
        print("Multinomial Naive Bayes Results:")
        mnb = MultinomialNB()
        mnb.fit(self.X_train, self.Y_train)
        print("Training Accuracy:", mnb.score(self.X_train, self.Y_train), end=", ")

        y_pred = mnb.predict(self.X_train)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_train, y_pred, average="macro")
        print("Precision (Training):", precision, end=", ")
        print("Recall (Training):", recall, end=", ")
        print("F1 (Training):", f1)

        print("\b")

        print("Test Accuracy:", mnb.score(self.X_test, self.Y_test), end=", ")
        y_pred_test = mnb.predict(self.X_test)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_test, y_pred_test, average="macro")
        print("Precision (Test):", precision, end=", ")
        print("Recall (Test):", recall, end=", ")
        print("F1 (Test):", f1)

warnings.filterwarnings("ignore")
print("Python version:", python_version())

if len(sys.argv) == 1:
    raise Exception("Must provide data filepath as input.")
else:
    url = sys.argv[1]

hw_class = run_hw(url)
hw_class.read_data()
hw_class.keep_review_ratings()
hw_class.labeling_reviews()
hw_class.select_random_rows()
hw_class.train_test_split()
hw_class.data_cleaning()
hw_class.pre_processing()
hw_class.tfidf()
hw_class.perceptron()
hw_class.svm()
hw_class.logistic_regression()
hw_class.naive_bayes()
