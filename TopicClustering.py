"""
(C) NerdGasm - 2016
Comment Summarization System for a Social Media Post using LDA based topic clustering and precedence based ranking.
"""

import numpy as np
import unicodedata
import lda.datasets
import re
import scipy.sparse as sp
import json
import codecs
import networkx as nx
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Text file containing top k comments for each cluster.
ranking_file = codecs.open("topComments.txt", "w", "utf-8")

# Text file containing the clustered comments.
cluster_file = codecs.open("clustering.txt", "w", "utf-8")

# Text file containing the list of summary comments for computing the retention rate.
summaryList_file = codecs.open("summaryList.txt", "w", "utf-8")

# Summary text file.
summary_file = codecs.open("summary.txt", "w", "utf-8")

# Text file containing the list of input comments for computing the retention rate.
input_file = codecs.open("input.txt", "w", "utf-8")

input_list = []
summary_list = []
original_comments = []

it_stop = get_stop_words('italian')
tokenizer = RegexpTokenizer(r'\w+')
c = CountVectorizer()

# Number of comments to Fetch within each cluster.
k = 3
# Json file containing the comments.
video_file = '/Users/sridharyadav/Downloads/SenTube/tablets_IT/video_-ps8odUxZpA-annotator:Agata.json'


def rankcomments(orig_commentcluster, commentcluster, k):
    """
    Rank comments in order to identify important and informative comments within each cluster using precedence based ranking.
    :param orig_commentcluster: Clustered comments preserving the stop words and special characters.
    :param commentcluster: Processed clustered comments.
    :param k: Number of comments to Fetch within each cluster
    """

    # Learn the vocabulary dictionary and return term-document matrix
    bow_matrix = c.fit_transform(commentcluster)
    # Learn vocabulary and idf, return term-document matrix.
    normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
    # Generate NetworkX adjacency graph
    similarity_graph = normalized_matrix * normalized_matrix.T
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    # Compute PageRank of the comments in the each cluster.
    scores = nx.pagerank(nx_graph, 0.85)
    # Rank the comments as per higher page rank.
    ranked = sorted(((scores[i], s) for i, s in enumerate(orig_commentcluster)), reverse=True)
    # Writing the top k comments for each cluster into a file.
    for tC in range(0, k):
        summary_list.append(ranked[tC][1])
        ranking_file.write("Comment {} : {}\n".format(tC, ranked[tC][1]))


def cluster_comments(doc_set, texts):
    """
    Clustering thematically-related comments through an application of topic-based clustering based on LDA
    :param doc_set: The list of all the comments.
    :param texts: Vocabulary of Italian words in the comments.
    :return: Clustered comments dictionary with key being the topic number and value being the list of all the
    comments in that topic.
    """

    # Convert the comments into to a matrix of token counts
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0)
    matrix = vectorizer.fit_transform(doc_set)
    # Generate the Compressed Sparse Row matrix
    matrix = sp.csr_matrix(matrix, dtype=np.int64, copy=False)
    vocab = tuple(texts)
    orig_titles = tuple(original_comments)
    titles = tuple(doc_set)
    # Implements latent Dirichlet allocation (LDA)
    model = lda.LDA(n_topics=3, n_iter=500, random_state=1)
    # Fit the model with matrix.
    model.fit(matrix)
    # Point estimate of the topic-word distributions.
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 8
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    # Point estimate of the document-topic distributions
    doc_topic = model.doc_topic_
    dictionary = defaultdict(list)
    orig_dictionary = defaultdict(list)
    for i in range(len(doc_set)):
        dictionary[int(doc_topic[i].argmax())].append(titles[i])
        orig_dictionary[int(doc_topic[i].argmax())].append(orig_titles[i])
    return dictionary,orig_dictionary


def read_data(video_file):
    """
    Read the comments from the Sentube corpus. The data is in the from of a json for each video.
    :param video_file: The video json file containg the comments
    :return: Set of all the comments for the video.
    """

    video = json.load(open(video_file))
    comment_set = []
    for comment in video['comments']:
        input_list.append(comment.get('text').replace("\n", " "))
        comment_set.append(re.sub('[^a-zA-Z0-9\s\P{P}\']+', r'', comment.get('text').replace("\n", " ")))
        original_comments.append(unicodedata.normalize('NFKD', comment.get('text')).encode('ascii','ignore'))
    return comment_set


def clean_data(comments_list):
    """
    Tokenize the comments and remove the stopwords.
    :param comments_list: List of all the comments for the video.
    :return: Processed Vocabulary of Italian words within all the comments.
    """
    comment_texts = []
    # loop through document list
    for i in comments_list:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in it_stop]
        # add tokens to list
        comment_texts = comment_texts + stopped_tokens
    return comment_texts


def main():
    doc_set = read_data(video_file)
    texts = clean_data(doc_set)
    clusters,orig_clusters = cluster_comments(doc_set, texts)
    for key, value in clusters.iteritems():
        cluster_file.write("(top topic: {}) - {}  \n".format(key, value))
        ranking_file.write("Top comments in topic are\n")
        rankcomments(orig_clusters[key],value, k)
    summaryList_file.write(str(summary_list))

    # Writing the entire summary to a file.
    for comment in summary_list:
        summary_file.write(comment)

    input_file.write(str(input_list))
    summaryList_file.close()
    input_file.close()
    cluster_file.close()
    ranking_file.close()
    summary_file.close()

main()
