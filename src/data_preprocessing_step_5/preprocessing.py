import numpy as np
import pandas as pd
import os
from keras.preprocessing import sequence
from keras.applications import VGG16
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import pickle


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


def encode_images_with_model_features(df, model,
                                      pickle_filename, verbose=False):
    """
    Runs through each item in a df with an image name column through a keras
    model to get features. It then saves those features in a pkl file and
    returns them if needed for later.

    paramters:
    --------------------------
    df (pandas DataFrame) -> A dataframe containing a 'image_name' column that
        contains names of images to be encoded.
    model (keras model) -> The VGG16 model that creates the feature prediction
    pickle_filename (str) -> the name of the pkl file to create
    verbose (bool) -> If true the output will have print statements every 100
        images

    returns:
    --------------------------
    encoded_images (dict) -> A dictionary where each encoding is the value and
        the image name is the key
    A pickle object is also created
    """
    encoded_images = {}
    i = 0
    for img in df['image_name']:
        encoded_images[img] = get_encoding(model, img)
        i += 1
        if i % 100 == 0 and verbose:
            print('Enoded: {} images.'.format(i))
    with open(pickle_filename, "wb") as pickle_f:
        pickle.dump(encoded_images, pickle_f)
    if verbose:
        print('pickle created as {}'.format(pickle_filename))
    return encoded_images


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


def pad_sequences(df, max_len, vocab_size, word_2_indices):
    """
    Takes in a series of descriptions from a dataframe, converts each
    description to an array, and then post pads the sequences with zeros until
    they reach their max length. Finally they are added to a list and converted
    to numpy arrays. At the same time we will create an array of next words
    which can be used as labels for the previous predictions.

    paramters:
    --------------------------
    df (pandas DataFrame) -> A dataframe with string based descriptions
    max_len (int) -> The number of words in the longest description
    vocab_size (int) -> How many unique words exisat in the data
    word_2_indices (dict) -> A dictionary with words as keys and indices
        as values

    returns:
    --------------------------
    padded_sequences (np array) -> An array of padded dequences
    subsequent_words (np array) -> An array with the following word encoded
    """
    padded_sequences, subsequent_words = [], []

    for ix in range(df.shape[0]):
        partial_seqs = []
        next_words = []
        text = df['description'][ix].split()
        text = [word_2_indices[i] for i in text]
        for i in range(1, len(text)):
            partial_seqs.append(text[:i])
            next_words.append(text[i])
        padded_partial_seqs = sequence.pad_sequences(partial_seqs,
                                                     max_len, padding='post')

        next_words_1hot = np.zeros([len(next_words), vocab_size],
                                   dtype=np.bool)

        for i, next_word in enumerate(next_words):
            next_words_1hot[i, next_word] = 1

        padded_sequences.append(padded_partial_seqs)
        subsequent_words.append(next_words_1hot)

    padded_sequences = np.asarray(padded_sequences)
    subsequent_words = np.asarray(subsequent_words)

    return padded_sequences, subsequent_words


def create_abbreviated_arrays_for_model(num_images):
    """
    Creates arrays to use in the model for training. Creates a caption array,
    a next word array, a image encoding array, and an image name array.

    paramters:
    --------------------------
    num_images (int) -> the number of images to train on

    returns:
    --------------------------
    None,
    A captions numpy array containing padded sequences of each description for
        each image is created
    A next word numpy array containing arrays for the next words is created
    An array containing the encoded image is create_dicts
    An array containing the image name is created
    """
    captions = []
    next_words = []
    for ix in range(num_images):
        captions.extend(padded_sequences[ix])
        next_words.extend(subsequent_words[ix])
        if ix % 100 == 0:
            print('Finished: {}'.format(ix))

    captions = np.array(captions)
    next_words = np.array(next_words)
    np.save("../../data/captions" + str(num_images) + ".npy", captions)
    np.save("../../data/next_words" + str(num_images) + ".npy", next_words)

    imgs = []
    for ix in range(df_train.shape[0]):
        imgs.append(train_encoded_images[df_train['image_name'][ix]])
    imgs = np.asarray(imgs)

    images = []
    for ix in range(num_images):
        for iy in range(padded_sequences[ix].shape[0]):
            images.append(imgs[ix])
        if ix % 100 == 0:
            print('Finished: {}'.format(ix))
    images = np.asarray(images)

    np.save("../../data/images" + str(num_images) + ".npy", images)

    image_names = []
    for ix in range(num_images):
        for iy in range(padded_sequences[ix].shape[0]):
            image_names.append(df_train['image_name'][ix])
        if ix % 100 == 0:
            print('Finished: {}'.format(ix))
    image_names = np.asarray(image_names)

    np.save("../../data/image_names" + str(num_images) + ".npy", image_names)


if __name__ == '__main__':
    images_dir = os.listdir("../../data/pics/")
    images_path = "../../data/pics/"

    train_path = '../../data/image_description_pair_8k_train.csv'
    test_path = '../../data/image_description_pair_8k_test.csv'

    number_of_images = 100
    # read_csv
    df_train = pd.read_csv(train_path)[:number_of_images]
    df_test = pd.read_csv(test_path)[:number_of_images]
    print("CSV's loaded!")
    # image preprocessing
    # load keras VGG16 model
    vgg = load_vgg16_model()
    print("VGG16 Model loaded!")
    train_encoded_images = encode_images_with_model_features(df_train, vgg,
        "../../data/train_encoded_images.pkl", verbose=True)
    print('All train images encoded!')
    test_encoded_images = encode_images_with_model_features(df_test, vgg,
        "../../data/test_encoded_images.pkl", verbose=True)
    print('All test images encoded!')
    # language preprocessing
    vocab_size, max_len, unique_words = instantiate_vocab_stats(df_train)

    word_2_indices, indices_2_word = create_dicts(unique_words)

    padded_sequences, subsequent_words = pad_sequences(df_train, max_len,
                                         vocab_size, word_2_indices)

    print('All sequences padded')

    create_abbreviated_arrays_for_model(number_of_images)

    print('Finished')
