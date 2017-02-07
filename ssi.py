#!/usr/bin/env python3
"""module doc"""

import numpy as np
import os
from gensim.models import Word2Vec
import jieba
import jieba.analyse
import json
import sys
import pickle

# Conventions:
#
#   D: int = num of words
#   N: int = num of features
#   q, d: documents

N = 100

def model():
    if hasattr(model, 'm') and model.m is not None:
        return model.m

    if os.path.exists('word2vec.model'):
        model.m = Word2Vec.load('word2vec.model')
        return model()

    with open('corpus.txt', 'r') as f:
        lines = f.readlines()

    def tokenize(t):
        return filter(lambda w: w != ' ', jieba.lcut(t.strip()))
    sentences = map(tokenize, lines)

    model.m = Word2Vec(sentences, size=N)
    model.m.init_sims(replace=True)
    model.m.save('word2vec.model')

    return model()


def normalize(mat):
    zero_norm = np.all(mat == 0, axis=1, keepdims=True)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[zero_norm] = 1
    return mat / norm


def bag_of_words(text):
    m = model()
    tags = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
    tags = [t for t in tags if t[0] in m.vocab]

    words   = [t[0] for t in tags]
    weights = [t[1] for t in tags]

    indices = [m.vocab[w].index for w in words]

    vocab_count = m.syn0.shape[0]
    vec = np.zeros(vocab_count)

    vec[indices] = weights

    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1
    vec /= norm

    return vec.reshape([1, -1])


def articles():
    if hasattr(articles, 'db'):
        return articles.db

    if os.path.exists('article_db.bin'):
        with open('article_db.bin', 'rb') as f:
            articles.db = pickle.load(f)
        return articles()

    with open('articles.json', 'r') as f:
        _articles = json.load(f)

    articles.db = dict()
    for article in _articles:
        # weighted_text = ' '.join([article['title']]  * 10 + [article['body']])
        weighted_text = article['title']
        bow = bag_of_words(weighted_text)
        articles.db[str(article['id'])] = {
            'title': article['title'],
            'body': article['body'],
            'bow': bow
        }

    with open('article_db.bin', 'wb') as f:
        pickle.dump(articles.db, f)

    return articles()



def ssi_similarity(m, q, d):
    """
    n0,n1: num of docs
    m: word2vec model (N x D)
    q: normalized b-o-w for doc q (n0 x N)
    d: normalized b-o-w for doc d (n1 x N)

    ssi is calc-ed by f(q,d) = q . (m . m' + I) . d'
                             = q . m . m' . d' + q . d'
    """

    return np.dot(q.dot(m), m.T.dot(d.T)) + q.dot(d.T)
    #return q.dot(d.T)


def find_similar(q_bow):
    q = q_bow
    m = normalize(model().syn0)
    arts = articles().values()
    d = np.array([a['bow'] for a in arts]).reshape(len(arts), -1)
    sim = ssi_similarity(m, q.reshape([1,-1]), d)
    ordered = sorted(zip(sim[0,:], arts),
                     key=lambda x: x[0],
                     reverse=True)

    return ordered


def main():
    action = sys.argv[1]

    if action == 'sim':
        q_id = sys.argv[2]
        q_bow = articles()[q_id]['bow']
        arts = find_similar(q_bow)

    elif action == 'kw':
        q_bow = bag_of_words(sys.argv[2])
        arts = find_similar(q_bow)

    else:
        print("unknown action")

    for sim, article in arts[:10]:
        print("{:<16}: {}".format(str(sim), article['title']))



if __name__ == '__main__':
    main()