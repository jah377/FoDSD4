#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  31 11:31:25 2021

@author: jonathanharris
"""

import re
import nltk
from text2digits import text2digits
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

set(stopwords.words('english'))


def clean_string(s):
    '''
    remove defined characters and substrings 

    Parameters
    ----------
    s : raw string

    Returns
    -------
    s : cleaned string

    '''
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
    '''
    Remove stopwords and short words from a clean, tokenized string

    Parameters
    ----------
    tokens : list of strings

    Returns
    -------
    tokens : list of strings wo stopwords

    '''

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


def normalize_string(tokens):
    '''
    Lemmatize tokens

    Parameters
    ----------
    tokens : list of strings

    Returns
    -------
    tokens_lemma : list of lemmatized strings

    '''

    lemma = WordNetLemmatizer()  # instantiate model

    return [lemma.lemmatize(token) for token in tokens]
