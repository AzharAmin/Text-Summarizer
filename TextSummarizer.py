import string
from nltk.corpus import stopwords
import numpy as np
import networkx as nx


from nltk.cluster import cosine_distance

name = input('Enter .txt file: ')
handle = open(name, 'r')
lines = handle.readlines()
handle = open(name, 'w')
mystr = ''.join([line.strip() for line in lines])+"\n"
handle.write(mystr)
handle.close()

handle = open(name, 'r')
wordlist = list()
counts = dict()
file = open(name, 'r')

for line in handle:
    line = line.translate(str.maketrans('', '', string.punctuation))
    line = line.lower()
    words = line.split()

    for word in words:
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1

    for word in words:
        if word in wordlist:
            continue
        wordlist.append(word)

wordlist.sort()
print("\n")
print(wordlist)
print("\n")

lst = list()
for key, val in list(counts.items()):
    lst.append((val, key))


lst.sort(reverse=True)

for key, val in lst[:10]:
    print(val.upper(), " is used ", key, "times")

print('\n')


def read_article(file):

    file = open(name, 'r')
    filedata = file.readlines()
    article = filedata[0].split(".")
    sentences = []

    for sentence in article:
        #print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences = read_article(file)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    print("Indexes of top ranked_sentence order are ", ranked_sentence)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - output the summarize text
    print("\nSummarized Text: \n", ". ".join(summarize_text))

# Result
generate_summary(" ", 2)
