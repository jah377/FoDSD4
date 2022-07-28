import re
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text2digits import text2digits

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
set(stopwords.words('english'))


def clean_string(s):
    """
    remove defined characters and substrings
    """
    t2d = text2digits.Text2Digits()  # instantiate model

    s = s.lower()  # normalize lower case
    s = re.sub('mr\.', ' ', s)  # remove Mr.
    s = re.sub('mrs\.', ' ', s)  # remove Mrs.
    s = re.sub('[\(\[].*?[\)\]]', '', s)  # remove references
    s = re.sub('[0-9]+.\t', '', s)  # remove paragraph numbers
    s = re.sub('\n ', ' ', s)  # remove new lines
    s = re.sub('\n', ' ', s)  # remove new lines
    s = re.sub("'s", ' ', s)  # remove apostrophe-s
    s = re.sub("'", '', s)  # remove apostrophe
    s = re.sub('-', ' ', s)  # remove hyphens
    s = re.sub('- ', ' ', s)  # remove hyphens
    s = re.sub("\'", '', s)  # remove quotations

    s = re.sub(r'[^a-zA-Z0-9 ]', '', s)  # remove punctuation/non-alpha
    s = t2d.convert(s)  # converts 'fifth' to 5 -- remove later
    s = re.sub(r'[0-9]', ' ', s)  # remove numbers from t2d
    s = re.sub(r'  +', ' ', s)  # remove 2+ spaces
    s = s.strip()  # remove leading-trailing spaces

    return s


def remove_stopwords(tokens):
    """
    remove stopwords and short words from clean, tokenized string
    """
    n_char = 3  # remove short words (arbitrary)
    sw = stopwords.words("english")  # stop words
    sw += ['assembly', 'community', 'continue', 'countries', 'country', 'general',
           'government', 'human', 'international', 'like', 'make', 'many', 'member',
           'must', 'nations', 'national', 'need', 'organization', 'organisation',
           'people', 'president', 'process', 'republic', 'region', 'secretary', 'session', 'state',
           'states', 'time', 'united', 'well', 'work', 'world', 'year']  # stop words determined from wordcloud
    sw = [*map(clean_string, sw)]  # reformat to match cleaned text

    # remove useless or short words
    tokens_wo_sw = [word for word in tokens
                    if (word not in sw) and len(word) > n_char]

    return tokens_wo_sw


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

    speeches = speeches.apply(
        lambda x: [w for w in x if w not in single_words])

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
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
            max_feature = tf_idf.columns[index - 1]
            words.append(max_feature)

        class_words_dict[classname] = words

    return class_words_dict


def display_words(words_dict: dict):
    """
    displays the top n most disctincive words
    """

    for key in words_dict:
        print(key)
        print(*words_dict[key], sep=", ")
        print('\n')
