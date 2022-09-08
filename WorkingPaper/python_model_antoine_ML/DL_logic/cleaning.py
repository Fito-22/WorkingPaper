import pandas as pd
import numpy as np

import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

'''
    USE ONLY THE LAST FUNCTION
    THAT COMBINES EVERYTHING
'''

def cutting_abs(text):
    '''
    Cutting everything before the abstract
    '''
    return text[text.find("ABSTRACT")+8:]

def lowercase(text):
    '''
    Put the text in lowercase
    '''
    return text.lower()

def remove_digit(text):
    '''
    Removing the digits
    '''
    cleaned_text = ''.join(char for char in text if not char.isdigit())
    return cleaned_text

def remove_punctuation(text):
    '''
    Removing the punctuation
    '''
    cleaned_text = ''.join(char for char in text if char not in string.punctuation)

    return cleaned_text

def try_with_regular_expression(text):
    '''
    Removing the last special characters
    using Regex
    '''
    new_string = re.sub(r"[^a-z]"," ",text)
    return new_string

def normalize_space_man(text):
    '''
    Normalize all the space in one character size
    '''
    lst_word = text.split()
    return " ".join(word for word in lst_word)

def remove_single(text):
    '''
    A lot of single characters word appears during
    prepro(due to mathematical formula) and we
    don't want them.
    '''
    lst_word = text.split()
    return " ".join(word for word in lst_word if len(word)>1)

def tokkenize_words(text):
    return word_tokenize(text)

def remove_stopwords(lst_word):
    stop_words = set(stopwords.words('english'))
    return [word for word in lst_word if not word in stop_words]

def lemmatize(lst_word):
    # Lemmatizing the verbs

    verb_lemmatized = [WordNetLemmatizer().lemmatize(word, pos = "v")  # v --> verbs
                  for word in lst_word]

    # 2 - Lemmatizing the nouns
    noun_lemmatized = [WordNetLemmatizer().lemmatize(word, pos = "n")  # n --> nouns
                  for word in verb_lemmatized]
    return noun_lemmatized

def get_text_back(lst_word):
    return " ".join(word for word in lst_word)


def final_cleaning(text:str)->str:
    '''
    the input is the dirty text
    The output is a cleaned text
    '''
    return get_text_back(\
                lemmatize(\
                    remove_stopwords(\
                        tokkenize_words(\
                            remove_single(\
                                normalize_space_man(\
                                    try_with_regular_expression(\
                                        remove_punctuation(\
                                            remove_digit(\
                                                lowercase(\
                                                    cutting_abs(text)))))))))))
