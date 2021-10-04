import re
import nltk
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('wordnet')


def remove_names(text: str):
    """
    removes all capitalized words
    """
    return " ".join([word for word in text.split() if word[0].islower()])


def lemmatizer(text: str):
    """
    lemmatizes the speeches, i.e. shortens words to its shortest
    form present in the dictionary
    """
    lm = WordNetLemmatizer()
    return [lm.lemmatize(w) for w in text]


def remove_single_occurence(speeches):
    """
    removes words that only occur once in the dataframe
    """
    cvectorizer = CountVectorizer(analyzer=lambda x: x).fit(speeches)
    cvectors = cvectorizer.transform(speeches).toarray()
    vocabulary = cvectorizer.vocabulary_

    # count how many times each word appears in whole dataset
    word_count = np.sum(cvectors, axis=0)

    # get index of words that only appear once
    single_indices = np.where(word_count == 1)

    # swap keys and values of vocabulary dict
    vocabulary = dict((v, k) for k, v in vocabulary.items())

    # get words that only appear once
    single_words = []
    for single_index in np.nditer(single_indices):
        single_word = vocabulary[int(single_index)]
        single_words.append(single_word)

    speeches = speeches.apply(lambda x: [w for w in x if w not in single_words])

    return speeches


def tf_idf(speeches):
    """
    computes the tf-idf for all the words in the speeches
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(speeches)
    df1 = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

    return df1


def logreg(X, y):
    """
    create a logistic regression model for classification
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    classes = model.classes_
    weights = model.coef_
    precision, recall, _, _ = precision_recall_fscore_support(y_test, pred)
    accuracy = accuracy_score(y_test, pred)

    return pred, classes, weights, accuracy, precision, recall


def top_n_distinctive_words(n, tf_idf, classes, weights):
    """
    returns the top n most distincive words per class for the logistic regression model
    """

    # feature weights of each class
    class_weight_dict = dict(zip(classes, weights))
    class_words_dict = dict()

    for classname, class_weights in class_weight_dict.items():

        # get words with highest weights per class
        max_indices = np.argpartition(class_weights, -n)[-n:]
        words = []

        for index in max_indices:
            max_feature = tf_idf.columns[index -1]
            words.append(max_feature)

        class_words_dict[classname] = words

    return class_words_dict


def display_words(words_dict: dict):
    """
    displays the top n most disctincive words
    """

    for key in words_dict:
        print(key)
        print(*words_dict[key], sep = ", ") 
        print('\n')
        
