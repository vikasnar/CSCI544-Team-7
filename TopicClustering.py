import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

# Json file containing the comments
comments_file = '/Users/sridharyadav/Downloads/SenTube/tablets_IT/video_-ps8odUxZpA-annotator:Agata.json'
it_stop = get_stop_words('italian')
tokenizer = RegexpTokenizer(r'\w+')

def read_data(comments):
    video = json.load(open(comments))
    comment_set = []
    for comment in video['comments']:
        comment_set.append(re.sub('[^a-zA-Z0-9\s\P{P}\']+', r'', comment.get('text').replace("\n", " ")))
        original_comments.append(unicodedata.normalize('NFKD', comment.get('text')).encode('ascii','ignore'))
    return comment_set


def clean_data(comments_list):
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
    doc_set = read_data(comments_file)
    texts = clean_data(doc_set)

main()
