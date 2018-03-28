import numpy as np
import pandas
import pickle
import re

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def load_data_labels(file_path, train=True):
    input_a = []
    input_b = []

    df = pandas.read_csv(file_path)
    qs1 = df['question1']
    qs2 = df['question2']
    if train:
        labels = df['is_duplicate']
        max_len = 0
        word_to_idx = {}
        word_to_idx["PAD"] = 0
        word_to_idx["unknown"] = 1
        # map every word to corresponding index
        for i, q1 in enumerate(qs1):
            words = text_to_word_list(q1)
            max_len = max(max_len, len(words))
            sentence = []
            for word in words:
                word = word.strip()
                if word not in word_to_idx:
                    idx = len(word_to_idx)
                    word_to_idx[word] = idx
                sentence.append(word_to_idx[word])
            input_a.append(sentence)

            q2 = qs2[i]
            words = text_to_word_list(q2)
            sentence = []
            max_len = max(max_len, len(words))
            for word in words:
                word = word.strip()
                if word not in word_to_idx:
                    idx = len(word_to_idx)
                    word_to_idx[word] = idx
                sentence.append(word_to_idx[word])
            input_b.append(sentence)
        # zero padding the sentences of which length is less than max length
        for idx in range(len(input_a)):
            sentence = input_a[idx]
            if len(sentence) < max_len:
                input_a[idx] = [0] * (max_len - len(sentence)) + sentence
            sentence = input_b[idx]
            if len(sentence) < max_len:
                input_b[idx] = [0] * (max_len - len(sentence)) + sentence
        f = open("../data/vocab.pickle", "wb")
        pickle.dump(word_to_idx, f)
        f.close()
        return [np.array(input_a), np.array(input_b), np.array(labels), word_to_idx]
    else:
        f = open("../data/vocab.pickle", "rb")
        word_to_idx = pickle.load(f)
        max_len = 0
        for i, q1 in enumerate(qs1):
            words = text_to_word_list(q1)
            max_len = max(max_len, len(words))
            sentence = []
            for word in words:
                word = word.strip()
                if word not in word_to_idx:
                    word = 'unknown'
                sentence.append(word_to_idx[word])
            input_a.append(sentence)
            q2 = qs2[i]
            words = text_to_word_list(q2)
            sentence = []
            for word in words:
                word = word.strip()
                if word not in word_to_idx:
                    word = 'unknown'
                sentence.append(word_to_idx[word])
            input_b.append(sentence)
        for idx in range(len(input_a)):
            sentence = input_a[idx]
            if len(sentence) < max_len:
                input_a[idx] = [0] * (max_len - len(sentence)) + sentence
            sentence = input_b[idx]
            if len(sentence) < max_len:
                input_b[idx] = [0] * (max_len - len(sentence)) + sentence

        return [np.array(input_a), np.array(input_b), word_to_idx]


def batch_iter(data, batch_size, num_epochs, train=True):
    """
    generate a batch iterator for a given dataset
    :param data: zip(x_train, y_train, sent_len_train, word_len_train)
    :param batch_size: batch size
    :param num_epoch: number of iteration
    :return: batch iterator
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if train:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
