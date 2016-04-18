"""
(C) NerdGasm - 2016
Evaluation of comment summarization system using Retention Rate measure.
A good summary should be short and yet retain as much as information of the input text as possible.
"""

from __future__ import division, unicode_literals
from textblob import TextBlob as tb
from stop_words import get_stop_words
import sys
import math

reload(sys)
sys.setdefaultencoding('utf8')


it_stop = get_stop_words('italian')

def tf(word, blob):
    """
    Compute tf of a word.
    :param word: Word whose tf is being computed
    :param blob: Comment
    :return: tf of the word.
    """
    return blob.words.count(word) / len(blob.words)

def n_containing(word, commentlist):
    """
    Compute the number of comments containing the word.
    :param word: Word whose idf is being computed
    :param commentlist: List of all the comments in the document.
    :return: Number of comments containing the word.
    """
    return sum(1 for blob in commentlist if word in blob)

def idf(word, commentlist):
    """
    Compute idf of a word.
    :param word: Word whose idf is being computed
    :param commentlist: List of all the comments in the document.
    :return: idf of the word.
    """
    return math.log(len(commentlist) / (1 + n_containing(word, commentlist)))

def tfidf(word, blob, commentlist):
    """
    Compute tf-idf of a word.
    :param word: Word whose tf-idf is being computed
    :param blob: Comment
    :param commentlist: List of all the comments in the document.
    :return: tf-idf of the word.
    """
    return tf(word, blob) * idf(word, commentlist)


def get_nonstop_words(text):
    """
    To generate the non-stop words for each comment.
    :param text: Comment
    :return: List of non-stop words for each comment.
    """
    tokens = text.split()
    stopped_tokens = [i for i in tokens if not i in it_stop]
    return stopped_tokens


def calculate_tfidf_average(commentlist):
    """
    Computing the average tf-idf score for the comments list in the document.
    :param commentlist: List of all the comments in the document.
    :return: The average tf-idf score for the comments list in the document.
    """
    scores = {}
    for i,blob in enumerate(commentlist):
        scores.update({word: tfidf(word, blob, commentlist) for word in blob.words})
    tf_idf_sum = 0
    for key in scores:
        tf_idf_sum += scores[key]
    return tf_idf_sum/len(scores)

def main():
    summary_text = eval(open("summaryList.txt").read())
    input_text = eval(open("input.txt").read())

    summarycommentlist = []
    inputcommentlist = []

    for comment in summary_text:
        summarycommentlist.append(tb(' '.join(get_nonstop_words(comment))))

    for comment in input_text:
        inputcommentlist.append(tb(' '.join(get_nonstop_words(comment))))

    tf_idf_summary = calculate_tfidf_average(summarycommentlist)
    tf_idf_input = calculate_tfidf_average(inputcommentlist)

    print "Retention Rate = ", tf_idf_summary/tf_idf_input

main()
