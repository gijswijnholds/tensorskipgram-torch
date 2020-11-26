import pickle


def dump_obj_fn(obj, fn):
    with open(fn, 'wb') as out_file:
        pickle.dump(obj, out_file)


def load_obj_fn(fn):
    with open(fn, 'rb') as in_file:
        data = pickle.load(in_file)
    return data


stopwords = set(['all', 'him', 'ourselves', 'no', 'being', "aren't", "doesn't",
                 'after', 'an', 'ours', 'y', 'themselves', 'up', 'she', 'does',
                 'where', 'wouldn', 'before', 'then', 'through', 'from',
                 'hadn', 'when', 'at', 'couldn', 'haven', 'while', 'll', 'did',
                 "hasn't", "it's", 'on', "couldn't", 'should', 'during',
                 'their', 'most', 'yours', 'how', 'you', 'yourselves',
                 "haven't", 'them', "shan't", 'hers', 'shouldn', 'under',
                 'off', 'whom', 'will', 'again', "needn't", 'than', "didn't",
                 'between', "mightn't", 'more', 'wasn', 'above', "wasn't",
                 'the', 'same', 'over', 'myself', 'our', 'doing', 'don', 'its',
                 'who', 'been', 'ain', 'can', 'mightn', 'against', 'both',
                 'because', 'am', 'what', 'only', 'won', 'is', 'any', 'some',
                 "you're", 'those', 'out', 'but', "weren't", 'herself', 'few',
                 'too', 'down', 'own', 'why', 'which', 'me', "mustn't", 'm',
                 'o', 'weren', 'himself', "wouldn't", 'cannot', 'hasn',
                 "you'll", 'into', 's', 'to', 'a', 'were', 'for', 'if', 'didn',
                 'needn', 'be', 're', 'aren', "hadn't", 'each', 'isn', 'with',
                 'there', 'as', 'further', 'they', 'shan', 'itself',
                 'yourself', 'it', 'other', 'and', 'here', "shouldn't", 'do',
                 'about', 'doesn', "she's", 'so', 'theirs', 'he', 'by', 't',
                 "that'll", 'just', 'such', "isn't", 'd', 'having', 'below',
                 'of', 'until', 'very', 'i', 'have', 'once', "should've", 'in',
                 'my', 'your', 'these', 'was', 'not', 'this', 'has', 'or',
                 'had', 'are', 'his', 've', "don't", 'ma', 'that', 'we', 'her',
                 'nor', "you've", "you'd", "won't", 'now', 'mustn'])
