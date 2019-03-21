import sys

import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction import stop_words
import xml.etree.cElementTree as ET
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import os
import numpy as np

PARTIALS = False



def gettext(xmltext):
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    xmltext = xmltext.encode('ascii', 'ignore') # ensure there are no weird char
    textString = ""
    root = ET.fromstring(xmltext)
    #Find the title tag, and add its text to the textString
    title = root.find('title')
    textString += title.text + " "

    #For every text tag, add its text to the textString
    for tag in root.iterfind('.//text/*'):
        textString += tag.text + " "

    return textString


def tokenize(text):
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3.
    """
    words = []
    tempWordList1 = []

    #Lowercase
    text = text.lower()

    #Remove all punctuation and numbers
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    text = regex.sub(" ", text)  # delete stuff but leave at least a space to avoid clumping together


    #Tokenize the test
    wordList = nltk.word_tokenize(text)

    #If a word is less than 3 characters,
    # don't move it forward
    for w in wordList:
        if len(w)>=3:
            tempWordList1.append(w)

    #If a word is in ENGLISH_STOP_WORDS
    #don't move it forward
    for w in tempWordList1:
        if w not in stop_words.ENGLISH_STOP_WORDS:
            words.append(w)

    return words




def stemwords(words):
    """
    Given a list of tokens/words, return a new list with each word
    stemmed using a PorterStemmer.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in words]


def tokenizer(text):
    return tokenize(text)


def compute_tfidf(corpus):
    """
    Create and return a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. The
    corpus argument is a dictionary mapping file name to xml text.
    """
    tfidf = TfidfVectorizer(input='content',
                            analyzer='word',
                            tokenizer=tokenizer,
                            stop_words='english',
                            decode_error='ignore')
    tfidf.fit(corpus)
    return tfidf


def summarize(tfidf, text, n):
    """
    Given a trained TfidfVectorizer object and some XML text, return
    up to n (word,score) pairs in a list.
    """
    #Create an iterable object for our single text
    corpus = {}
    corpus[0]=text

    #Transform the TFIDF vector
    X = tfidf.transform(corpus.values())

    #summarizeTerms are the terms in the document
    summarizeTerms = tokenizer(gettext(text))
    #terms are the terms in the tfidf matrix
    terms = tfidf.get_feature_names()
    #summarizeData is a middleman for the (word,score) tuples
    summarizeData = []

    #For every word in the document
    for word in summarizeTerms:
        #If its in the tfidf vocab
        if word in terms:
            #Grab the index
            termIndex = terms.index(word)
            #If its not already in summarizeData, add it
            if (word, X[0,termIndex]) not in summarizeData:
                summarizeData.append((word, X[0,termIndex]))

    #Scores is a sorted summarizeData
    #This was taken from test_tfidf
    scores = sorted(summarizeData, key=lambda item: f"{item[1]:.3f} {item[0]}", reverse=True)

    #Result is the top 20, filter >0.09
    result = []
    for i in range(0,n):
        if (scores[i][1] > 0.09):
            result.append((scores[i][0],scores[i][1]))

    return result

def load_corpus(zipfilename):
    """
    Given a zip file containing root directory reuters-vol1-disk1-subset
    and a bunch of *.xml files, read them from the zip file into
    a dictionary of (word,xmltext) associations. Use namelist() from
    ZipFile object to get list of xml files in that zip file.
    Convert filename reuters-vol1-disk1-subset/foo.xml to foo.xml
    as the keys in the dictionary. The values in the dictionary are the
    raw XML text from the various files.
    """

    zipDict = {}

    with zipfile.ZipFile(zipfilename,'r') as myzip:
        filelist = myzip.namelist()
        filelist.remove(filelist[0])

        #For every file in the list
        for file in filelist:
            #Open it
            #with myzip.open(str(file), 'r') as openfile:
                #Read and append text to zipDict
            xmltext = myzip.read(file).decode()
            zipDict[os.path.basename(file)] = xmltext

    return zipDict


