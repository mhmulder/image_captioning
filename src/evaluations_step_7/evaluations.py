import numpy as np
import os

import pandas as pd
from keras.preprocessing import image, sequence
from keras.applications import VGG16
from keras.models import Model
from keras.optimizers import Nadam
from PIL import Image
from IPython.display import display

from keras.layers import *
from keras.callbacks import TensorBoard
from scipy.spatial import distance
import spacy
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

"""
This script builds the net and trains the model. Alot of the functions could be
imported from preprocessing but for visualizing the whole enchilada I'll add
them to this script.
"""
nlp = spacy.load('en')


def instantiate_vocab_stats(df):
    """
    Based on the 'description' column of a DataFrame generate statistics
    required to create and process the data and net.

    paramters:
    --------------------------
    df (pandas DataFrame) -> A dataframe with string based descriptions

    returns:
    --------------------------
    vocab_size (int) -> How many unique words exisat in the data
    max_len (int) -> The number of words in the longest description
    unique (list) -> A sorted list of unique words in the data set
    """
    sentences = list(df['description'])
    tokens = [sentence.split() for sentence in sentences]
    words = [token for sublist in tokens for token in sublist]
    unique = sorted(list(set(words)))
    vocab_size = len(unique)
    max_len = 0
    for i in sentences:
        i = i.split()
        if len(i) > max_len:
            max_len = len(i)
    return vocab_size, max_len, unique


def create_dicts(unique_words):
    """
    Creates both a dict and a reverse dict mapping indices to words

    paramters:
    --------------------------
    unique_words (list) -> A list of unique words

    returns:
    --------------------------
    word_2_indices (dict) -> A dictionary with words as keys and indices
        as values
    indices_2_word (dict) -> A dictionary with indices as keys and words
        as values
    """
    word_2_indices = {val: index for index, val in enumerate(unique_words)}
    indices_2_word = {index: val for index, val in enumerate(unique_words)}

    return word_2_indices, indices_2_word


def load_vgg16_model():
    """
    Loads the VGG16 model from keras and truncates the last two layers.

    paramters:
    --------------------------
    None, the model is loaded into memory

    returns:
    --------------------------
    vgg (keras model) -> The VGG16 Model
    """
    vgg16 = VGG16(weights='imagenet', include_top=True,
                  input_shape=(224, 224, 3))
    vgg = Model(inputs=vgg16.input, outputs=vgg16.layers[-2].output)
    return vgg


def image_preprocess(img):
    """
    Use Keras built in image processing for VGG16, basically converts the image
    to an array, switches from RGB to BGR, and centers the mean

    paramters:
    --------------------------
    img (str) -> The name of an image

    returns:
    --------------------------
    imag (np array) -> A numpy array of size (3, 244, 244),  that is ready for
        VGG16 encoding
    """
    imag = load_img(img, target_size=(224, 224))
    imag = img_to_array(imag)
    imag = imag.reshape((1, imag.shape[0], imag.shape[1], imag.shape[2]))
    imag = preprocess_input(imag)
    return imag


def get_encoding(model, img):
    """
    Applies the VGG16 encoding to an img, (basically just returns a prediction)

    paramters:
    --------------------------
    model (keras model) -> The VGG16 model that creates the feature prediction
    img (np array) -> A preprocessed numpy array ready for prediction

    returns:
    --------------------------
    features (np array) -> A numpy array containing the predicted features of
        the image.
    """
    imag = image_preprocess(images_path + img)
    prediction = model.predict(imag)
    features = np.reshape(prediction, prediction.shape[1])
    return features


def build_model(embedding_size, max_len, vocab_size, load_weight_file=False):
    """
    Builds the net atchitecture and loads weights if specified.

    paramters:
    --------------------------
    embedding_size (int) -> Determines the number of nodes used for embedding
        text
    max_len (int) -> The number of words in the longest description
    vocab_size (int) -> How many unique words exisat in the data
    load_weight_file(str) -> If specified, the filepath for the weights should
        be entered and will be loaded for the model

    returns:
    --------------------------
    model (keras model) -> The model created by this function
    """
    embedding_size = embedding_size
    input_1 = Input(shape=(4096, ), dtype='float32')

    top_layers = Dense(embedding_size, activation='relu')(input_1)
    dropout_1 = Dropout(0.1)(top_layers)
    repeat_layer = RepeatVector(max_len)(dropout_1)

    input_2 = Input(shape=(max_len, ), dtype='float32')
    lang_embed = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=max_len)(input_2)
    lstm_1 = LSTM(256, return_sequences=True)(lang_embed)
    tdd = TimeDistributed(Dense(embedding_size))(lstm_1)


    merged = Concatenate()([repeat_layer, tdd])

    merged = Bidirectional(LSTM(1024, return_sequences=False))(merged)
    merged = Dropout(0.1)(merged)
    merged = Dense(vocab_size)(merged)
    predictions = Activation('softmax')(merged)

    model = Model(inputs=[input_1, input_2], outputs=predictions)
    if load_weight_file:
        model.load_weights(load_weight_file)

    model.compile(loss='categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])
    model.summary()

    return model


def cosine_sim_test(image_name, df, vgg, verbose = False):

    test_img = get_encoding(vgg, image_name)

    predicted_caption = argmax_pred_caption(test_img)
    clean_caption = df[df['image_name']==image_name]['description'].values[0]
    real_caption = df[df['image_name']==image_name]['real_description'].values[0]

    predicted_caption = stemmer(predicted_caption)
    clean_caption = stemmer(clean_caption)
    real_caption = stemmer(real_caption)


    clean_caption = clean_caption.split(" ")[1:-1]
    if verbose: print ('clean: ', " ".join(clean_caption))
    real_caption = real_caption.lower().split(" ")
    if verbose: print ('real: ', " ".join(real_caption))
    predicted_caption = predicted_caption.split(" ")
    if verbose: print ('pred: ', " ".join(predicted_caption))

    clean_index = np.zeros((len(word_2_indices),))
    for elem in clean_caption:
        index = int((word_2_indices.get(elem,0)))
        clean_index[index] = 1


    real_index = np.zeros((len(word_2_indices),))
    for elem in real_caption:
        index = int((word_2_indices.get(elem,0)))
        real_index[index] = 1

    predicted_index = np.zeros((len(word_2_indices),))
    for elem in predicted_caption:
        index = int((word_2_indices.get(elem,0)))
        predicted_index[index] = 1

    pred_clean_cosin = distance.cosine(clean_index,predicted_index)
    pred_real_cosin = distance.cosine(real_index,predicted_index)
    if verbose: print ("predicted vs clean cosine score: ", pred_clean_cosin)
    if verbose: print ("predicted vs real cosine score: ", pred_real_cosin)
    return pred_clean_cosin, pred_real_cosin

def cosine_eval(test_images, df, return_ones = False):
    cosine_score_0 = []
    cosine_score_1 = []
    return_ones_list = []
    for image_name in test_images:
        cos_0, cos_1 = cosine_sim_test(image_name, df)
        cosine_score_0.append(cos_0)
        cosine_score_1.append(cos_1)
        if return_ones and cos_0 == 1:
            return_ones_list.append(image)

    pred_clean_score = np.array(cosine_score_0)
    pred_real_score = np.array(cosine_score_1)
    print ("predicted vs clean cosine score: ", pred_clean_score.mean())
    print ("predicted vs real cosine score: ", pred_real_score.mean())
    if return_ones:
        return pred_clean_score, pred_real_score, return_ones_list
    return pred_clean_score, pred_real_score

def human_eval(test_images , df):
    # only runs in jupyter as of now
    scores = []
    for image_name in test_images:
        img = Image.open(images_path+image)
        test_img = get_encoding(vgg, image_name)

        z = img.resize((300, 300), Image.ANTIALIAS)
        display(z)

        predicted_caption = argmax_pred_caption(test_img)
        real_caption = df[df['image_name']==image_name]['real_description'].values[0]

        real_caption = set(real_caption.lower().split(" "))
        print ('Real: ', " ".join(real_caption))
        predicted_caption = set(predicted_caption.split(" "))
        print ('Prediction: ', " ".join(predicted_caption))

        score = int(input("On a scale of 0-5 how relevant are the words in the picture?\n >>>"))

        scores.append(score)
    scores = np.array(scores)
    print ('The mean score is: ', scores.mean())

    return scores

def argmax_pred_caption(image):
    caption = ["<start>"]
    while True:
        prediction_idx = [word_2_indices[word] for word in caption]
        padded_sequence = sequence.pad_sequences([prediction_idx], maxlen=max_len, padding = 'post')
        im_arr = np.array([image])
        pad_seq_arr = np.array(padded_sequence)
        prediction = model.predict([im_arr, pad_seq_arr])
        top_index = np.argmax(prediction[0], axis = 0)
        top_word = indices_2_word[top_index]
        caption.append(top_word)

        if top_word == '<end>' or len(caption) >= max_len:
            break
    return ' '.join(caption[1:-1])

def beam_search_predictions(image, beam_index = 3):
    start = [word_2_indices["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            preds = model.predict([np.array([image]), np.array(par_caps)])

            word_preds = np.argsort(preds[0])[-beam_index:] #Top n prediction

            for w in word_preds: #new list so as to feed it to model again
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:] # Top n words

    start_word = start_word[-1][0]
    intermediate_caption = [indices_2_word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

def stemmer(string):
    new_string = []
    for token in nlp(string):
        new_string.append(token.lemma_)
    return " ".join(new_string)

if __name__ == '__main__':
    images_dir = os.listdir("../../data/pics/")
    images_path = "../../data/pics/"

    vgg = load_vgg16_model()

    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=64, write_graph=True, write_images=False)

    train_path='../../data/image_description_pair_20k_train.csv'
    test_path='../../data/image_description_pair_20k_test.csv'

    df_train = pd.read_csv(train_path)

    vocab_size, max_len, unique_words = instantiate_vocab_stats(df_train)

    word_2_indices, indices_2_word = create_dicts(unique_words)

    captions = np.load("../../data/captions20k.npy")
    next_words = np.load("../../data/next_words20k.npy")
    images = np.load("../../data/images20k.npy")
    image_names = np.load("../../data/image_names20k.npy")

    # images is already 32 bit, otherwise convert things to 32 to run faster
    # on those super sweet GPU instances
    captions32 = captions.astype('float32')
    next_words32 = next_words.astype('float32')

    embedding_size = 256
    model = build_model(embedding_size, max_len, vocab_size, load_weight_file=False)

    # If you get an error about insize on inputs being incorrect make sure all file paths are correct and matching from previous scripts
    model.fit([images, captions32], next_words32, batch_size=512, validation_split = 0.2, epochs=9, callbacks=[tbCallBack])

    model.save("../../data/final_model.h5")
