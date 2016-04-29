from stop_words import get_stop_words
import itertools
import re
import numpy as np
from collections import OrderedDict


it_stop = get_stop_words('italian')
word_count_summary = dict()
word_count_comments = dict()
summary_prob_dist = OrderedDict()
comments_prob_dist = OrderedDict()
vocab = []


def get_word_count(comments_list, count_dict):
    """
    To get the probability of each word in the comment
    :param comments_list: List of tokens excluding the stop words in a comment.
    :param count_dict: Stores count of each word in a comment.
    :return : Dict containing count of each word in a comment.
    """
    for word in comments_list:
        word = re.sub("[\.\t\,\:!?;\(\)\.]", "", word, 0, 0)
        if word not in vocab:
            vocab.append(word)
        if word in count_dict:
            count_dict[word] += 1
        else:
            count_dict[word] = 1
    return count_dict


def get_probability_distribution(count_dict, prob_dict):
    """
    To get the probability of each word in the comment
    :param count_dict: Count of each word in the comment
    :param prob_dict: Stores probabilities of each word.
    :return : Dict containing probabilities of each word in a comment.
    """
    count = sum(count_dict.values())
    for word in count_dict:
        if word not in prob_dict:
            prob_dict[word] = float(count_dict[word]+1) / float(count + len(vocab));
    return prob_dict


def get_nonstop_words(text):
    """
    To generate the non-stop words for each comment.
    :param text: Comment
    :return: List of non-stop words for each comment.
    """
    tokens = text.split()
    stopped_tokens = [i for i in tokens if not i in it_stop]
    return stopped_tokens


def normalize(token_dict):
    """
    To normalize the probability distributions
    :param token_dict: The dict to normalize
    :return: Normalized distribution
    """
    for word in vocab:
        if word not in token_dict:
            token_dict[word] = float(1)/len(vocab)
    return token_dict


def kl(p, q):
    """
    Calculate KL entropy score between probability distributions
    :param p: Discrete probability distribution.
    :param q: Discrete probability distribution.
    :return: Calculated entropy
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def main():
    summary_text = eval(open("summaryList.txt").read())
    input_text = eval(open("input.txt").read())

    summarycommentlist = []
    inputcommentlist = []

    for comment in summary_text:
        summarycommentlist.append(get_nonstop_words(comment))

    for comment in input_text:
        inputcommentlist.append(get_nonstop_words(comment))

    get_probability_distribution(get_word_count(list(itertools.chain(*summarycommentlist)), word_count_summary),
                                 summary_prob_dist)

    get_probability_distribution(get_word_count(list(itertools.chain(*inputcommentlist)), word_count_comments),
                                 comments_prob_dist)

    summary_vector = list(normalize(summary_prob_dist).values())
    comments_vector = list(normalize(comments_prob_dist).values())

    print kl(summary_vector, comments_vector)


main()
