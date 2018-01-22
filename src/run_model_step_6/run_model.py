import numpy as np
import os
import pandas as pd
from keras.applications import VGG16
from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import *
from keras.callbacks import TensorBoard
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

"""
This script builds the net and trains the model. Alot of the functions could be
imported from preprocessing but for visualizing the whole enchilada I'll add
them to this script.
"""


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
    lang_embed = Embedding(output_dim=embedding_size, input_dim=vocab_size,
                           input_length=max_len)(input_2)
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

    model.compile(loss='categorical_crossentropy', optimizer=Nadam(),
                  metrics=['accuracy'])
    print(model.summary())

    return model

if __name__ == '__main__':
    images_dir = os.listdir("../../data/pics/")
    images_path = "../../data/pics/"

    vgg = load_vgg16_model()

    tbCallBack = TensorBoard(log_dir='../../data/logs', histogram_freq=64,
                             write_graph=True, write_images=False)

    train_path = '../../data/image_description_pair_20k_train.csv'
    test_path = '../../data/image_description_pair_20k_test.csv'

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
    model = build_model(embedding_size, max_len, vocab_size,
                        load_weight_file=False)

    # If you get an error about insize on inputs being incorrect make sure all
    # file paths are correct and matching from previous scripts
    model.fit([images, captions32], next_words32, batch_size=512,
              validation_split=0.2, epochs=9, callbacks=[tbCallBack])

    model.save("../../data/final_model.h5")

    # evaluate to your hearts content
