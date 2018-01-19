
import os
import string
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.model_selection import train_test_split

PATH = "data/"

ps = PorterStemmer()

lamma = WordNetLemmatizer()

#split data for testing and training
def split_data(views):
    return train_test_split(views, test_size=0.20, random_state=33)


def lemmatzing(words):
    clean_words = []
    for word in words:
        clean_words.append(lamma.lemmatize(word))
    return clean_words


def sent_token(text):
    return sent_tokenize(text)


def word_token(text):
    return word_tokenize(text)


def steamming(words):
    clean_words = []
    for word in words:
        clean_words.append(ps.stem(word))
    return clean_words


def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


def remove_puncatuation(words):
    return [word for word in words if word.lower() not in string.punctuation]


def remove_number(words):
    clean_word = []
    for word in words:
        if word not in string.digits:
            clean_word.append(word)
    return clean_word


def clean_data(sents, labs):
    i = 0
    views = []
    while len(labs) > i:
        if labs[i] == "0":
            views.append((create_word_features(remove_number(remove_puncatuation(word_tokenize(sents[i])))), "negative"))
        if labs[i] == "1":
            views.append((create_word_features(remove_number(remove_puncatuation(word_tokenize(sents[i])))), "positive"))

        i += 1
    return views


def create_data():
    sentences = []  # for storing sentences for training and testing
    labels = []  # for storing labels(Positive or negative) for training and testing

    fileName = [x for x in os.listdir(PATH)]

    for file in fileName:
        # get absulate file path
        path = os.path.join(PATH, file)
        # open file and store in file variable
        file = open(path, "r")
        # read the file text file save in sents variable
        sents = file.read().lower()
        # converts sents in sent_tokenizer and save respective variable
        sents = sent_token(sents)
        i = 0
        while len(sents) > i:
            if i == 0:
                obj1 = sents[i]
                obj2 = sents[i + 1]

                sentence = obj2.split("\n")
                sentences.append(obj1)
                labels.append(sentence[0])

            elif i == len(sents) - 1:
                obj1 = sents[i]
                obj2 = sents[i - 1]

                sentence = obj2.split("\n")

                sentences.append(sentence[1])
                labels.append(obj1[0])
            else:
                obj = sents[i]
                obj2 = sents[i + 1]

                sentence = obj.split("\n")
                label = obj2.split("\n")

                sentences.append(sentence[1])
                labels.append(label[0])

            i += 1

    print("Sentences:-", len(sentences))
    print("Labels:-", len(labels))
    return sentences, labels


if __name__ == "__main__":
    sentences, labels = create_data()

    views = clean_data(sentences, labels)

    train, test = split_data(views)

    classif = NaiveBayesClassifier.train(train)

    example1 = "Cats are awesome!"

    example2 = "I don’t like cats."

    example3 = "I have no headache!"

    example4 = "I hate dogs."

    print("%s :- %s" % (example1, classif.classify(create_word_features(word_token(example1)))))
    print("%s :- %s" % (example2, classif.classify(create_word_features(word_token(example2)))))
    print("%s :- %s" % (example3, classif.classify(create_word_features(word_token(example3)))))
    print("%s :- %s" % (example4, classif.classify(create_word_features(word_token(example4)))))

    print("Accuracy:-", accuracy(classif, test))
