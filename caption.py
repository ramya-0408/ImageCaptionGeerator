
import os
import argparse
import numpy as np
import streamlit as st
import tensorflow as tf
from pickle import load
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


def greedySearch(photo, max_length=34):
    global model, wordtoix, ixtoword
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def get_predictions():
    global images, encoding_test
    index = np.random.choice(1000)
    pic = list(encoding_test.keys())[index]
    image = encoding_test[pic].reshape((1, 2048))
    x = plt.imread(images+pic)
    # np.resize(x, (-1, 200, 350))
    st.sidebar.image(x, use_column_width=True)
    caption = greedySearch(image)
    return caption


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This app predicts image captions.')
    parser.add_argument('--modelfile',
                        help='File Path for trained model text.')
    parser.add_argument('--textdir',
                        help='File Path for Flickr dataset text.')
    parser.add_argument('--picklefile',
                        help='File Path for encoded pickle.')
    parser.add_argument('--imagedir',
                        help='Root directory path for Flickr dataset images.')
    parser.add_argument('--descriptions',
                        help='File Path for descriptions.')
    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)

    filename = args.textdir
    train = load_set(filename)

    # descriptions
    train_descriptions = load_clean_descriptions(args.descriptions, train)

    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)

    word_count_threshold = 10
    word_counts = dict()
    ixtoword = dict()
    wordtoix = dict()
    nsents = 0
    ix = 1

    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    model = tf.keras.models.load_model(args.modelfile)

    images = args.imagedir

    with open(args.picklefile, "rb") as encoded_pickle:
        encoding_test = load(encoded_pickle)

    st.title('Image caption generator')
    st.sidebar.markdown('## Input Image')

    if st.button('Generate a Random Image'):
        caption = get_predictions()
    else:
        caption = ''
    st.sidebar.markdown('## Caption Generated is: ')
    st.sidebar.markdown(caption)
main_modified.py
Displaying main_modified.py. 
