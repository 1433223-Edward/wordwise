import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import time

words = ["created", "has", "tested","bitterly","basics","basically","was","worker","farmer","trees"]

wnl = WordNetLemmatizer()


def get_word_category(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.VERB


for i in range(len(words)):
    print(words[i] + '-->' + wnl.lemmatize(words[i], get_word_category(pos_tag([words[i]])[0][1])))

print(time.time())