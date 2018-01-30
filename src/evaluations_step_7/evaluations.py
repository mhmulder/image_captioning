import numpy as np
import os
import pandas as pd
from keras.preprocessing import sequence
from keras.applications import VGG16
from keras.models import Model, load_model
from PIL import Image
from IPython.display import display
from keras.layers import *
from scipy.spatial import distance
import spacy
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
This script contains many of gthe functions I used to make predictions and
evaluate the model.

Some of these evaluation scripts were written to be viewed in a jupyter
notebook, as a result some of the dependencies have been commented out. For
full functionality of the human evaluator please import these scripts into a
notebook with a gui.
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


def get_encoding(model, img, default_direct=True):
    """
    Applies the VGG16 encoding to an img, (basically just returns a prediction)

    paramters:
    --------------------------
    model (keras model) -> The VGG16 model that creates the feature prediction
    img (np array) -> A preprocessed numpy array ready for prediction
    default_direct (bool) -> If true the image_path is automatically appended
    returns:
    --------------------------
    features (np array) -> A numpy array containing the predicted features of
        the image.
    """
    if default_direct:
        imag = image_preprocess(images_path + img)
    else:
        imag = image_preprocess(img)
    prediction = model.predict(imag)
    features = np.reshape(prediction, prediction.shape[1])
    return features


def argmax_pred_caption(filepath):
    """
    Uses argmax to make a description prediction

    paramters:
    --------------------------
    filepath (str) -> The filepath for the photo

    returns:
    --------------------------
    (str) -> The caption generated
    """
    image = get_encoding(vgg, filepath)
    prediction_idx = [1] # ! is index for <start>
    while True:
        padded_sequence = sequence.pad_sequences([prediction_idx],
                                                 maxlen=max_len,
                                                 padding='post')
        result = model.predict([np.array([image]), padded_sequence])
        sorted_results = np.argsort(result)
        top_result = sorted_results[0][-1]
        prediction_idx.append(top_result)
        if top_result == 0 or len(prediction_idx) >= max_len:
            break
    return (" ".join([indices_2_word[idx] for idx in prediction_idx[1:-1]]))


def beam_search_decoder(filepath, k=3):
    """
    Uses beam search with beam index of k to make a description prediction

    paramters:
    --------------------------
    filepath (str) -> The filepath for the photo
    k (int) -> beam index, top n predictions to use

    returns:
    --------------------------
    (str) -> The caption generated
    """
    image = get_encoding(vgg, filepath)
    prediction_idx = [[[1], 0.0]] # 1 is index for <start>
    while len(prediction_idx[0][0])< max_len:
        candidates = []
        for i in range (len(prediction_idx)):
            seq = prediction_idx[i][0]
            padded_sequence = sequence.pad_sequences([seq],
                                                 maxlen=max_len,
                                                 padding='post')
            result = model.predict([np.array([image]), padded_sequence])
            sorted_results = np.argsort(result[0])[-k:]
            top_k_result = sorted_results[-k:]
            top_k_probs = result[0][top_k_result]
            for n in range (k):
                new_seq = prediction_idx[i][0].copy()
                score = prediction_idx[i][1]

                new_seq.append(top_k_result[n])
                score = prediction_idx[i][1]
                new_score = score + top_k_probs[n]
                candidates.append([new_seq, new_score])
        prediction_idx = candidates
        prediction_idx = sorted(prediction_idx, reverse=False, key=lambda tup:tup[1])
        prediction_idx = prediction_idx[-k:]
    prediction_idx = prediction_idx[-1][0]

    final_description = []

    for i in prediction_idx:
        if i != 0:
            final_description.append(i)
        else:
            break
    final_words = [indices_2_word[i] for i in final_description]
    final_words = ' '.join(final_words[1:])
    return final_words


def cosine_sim_test(image_name, df, vgg, verbose=False):
    """
    For a given image, compares the predicted caption against the clean caption
    and real caption using an argmax prediction.

    paramters:
    --------------------------
    image_name (str) -> The filepath for the photo
    df (pandas Dataframe) -> containging an image_name, and corresponding
        description and real_description columns
    vgg (keras model) -> the VGG16 model
    verbose (bool) -> If true the function prints out the results

    returns:
    --------------------------
    pred_clean_cosin (float) -> the cosine score comparing the predicted
        description to the cleaned description
    pred_real_cosin (float) -> the cosine score comparing the predicted
        description to the real description
    """

    predicted_caption = argmax_pred_caption(image_name)
    clean_caption = df[df['image_name'] == image_name]['description'].values[0]
    real_cap = df[df['image_name'] == image_name]['real_description'].values[0]

    predicted_caption = lemmatizer(predicted_caption)
    clean_caption = lemmatizer(clean_caption)
    real_cap = lemmatizer(real_cap)

    clean_caption = clean_caption.split(" ")[1:-1]
    if verbose:
        print('clean: ', " ".join(clean_caption))
    real_cap = real_cap.lower().split(" ")
    if verbose:
        print('real: ', " ".join(real_cap))
    predicted_caption = predicted_caption.split(" ")
    if verbose:
        print('pred: ', " ".join(predicted_caption))

    clean_index = np.zeros((len(word_2_indices),))
    for elem in clean_caption:
        index = int((word_2_indices.get(elem, 0)))
        clean_index[index] = 1

    real_index = np.zeros((len(word_2_indices),))
    for elem in real_cap:
        index = int((word_2_indices.get(elem, 0)))
        real_index[index] = 1

    predicted_index = np.zeros((len(word_2_indices),))
    for elem in predicted_caption:
        index = int((word_2_indices.get(elem, 0)))
        predicted_index[index] = 1

    pred_clean_cosin = distance.cosine(clean_index, predicted_index)
    pred_real_cosin = distance.cosine(real_index, predicted_index)
    if verbose:
        print("predicted vs clean cosine score: ", pred_clean_cosin)
    if verbose:
        print("predicted vs real cosine score: ", pred_real_cosin)
    return pred_clean_cosin, pred_real_cosin


def cosine_eval(test_images, df, return_ones=False):
    """
    Runs the cosin_sim_Test for multiple images at a time and returns the mean
    score

    paramters:
    --------------------------
    test_images (list) -> A list of photos
    df (pandas Dataframe) -> Containging an image_name, and corresponding
        description and real_description columns
    return_ones (bool) -> If true returns the images where cosine sim = 1

    returns:
    --------------------------
    pred_clean_score (float) -> The  mean cosine score comparing the predicted
        descriptions to the cleaned descriptions
    pred_real_score (float) -> The mean cosine score comparing the predicted
        descriptions to the real descriptions
    return_ones_list (list) -> A list of images whose cosine similarity score
        is equal to one (bad fits)
    """

    cosine_score_0 = []
    cosine_score_1 = []
    return_ones_list = []
    for image_name in test_images:
        cos_0, cos_1 = cosine_sim_test(image_name, df)
        cosine_score_0.append(cos_0)
        cosine_score_1.append(cos_1)
        if return_ones and cos_0 == 1:
            return_ones_list.append(image_name)

    pred_clean_score = np.array(cosine_score_0)
    pred_real_score = np.array(cosine_score_1)
    print("predicted vs clean cosine score: ", pred_clean_score.mean())
    print("predicted vs real cosine score: ", pred_real_score.mean())
    if return_ones:
        return pred_clean_score, pred_real_score, return_ones_list
    return pred_clean_score, pred_real_score


def human_eval(test_images):
    # only runs in jupyter as of now
    """
    takes a list of images and displays its predicted caption and the real
    caption. It then allows the user to input a score between 0 and 5. At the
    end of the evaluation it prints the mean score.

    paramters:
    --------------------------
    test_images (list) -> A list of photos
    df (pandas Dataframe) -> Containging an image_name, and corresponding
        description and real_description columns

    returns:
    --------------------------
    scores (np array) -> A list of user evaluated scores.
    """
    scores = []
    for image_name in test_images:
        img = Image.open(images_path+image_name)
        z = img.resize((300, 300), Image.ANTIALIAS)
        display(z)

        predicted_caption = argmax_pred_caption(image_name)
        print('Prediction: ', predicted_caption)

        score = int(input("On a scale of 0-5 how relevant are the words" +
                          "in the picture?\n >>>"))

        scores.append(score)
    scores = np.array(scores)
    print('The mean score is: ', scores.mean())
    return scores


def lemmatizer(string):
    """
    Uses spaCy to lemmatize the words so that when doing the cosine similarity
    only the lemmatized version of the word is used.

    paramters:
    --------------------------
    string (str) -> A description

    returns:
    --------------------------
    (str) -> A lemmatized copy of the string
    """
    new_string = []
    for token in nlp(string):
        new_string.append(token.lemma_)
    return " ".join(new_string)


if __name__ == '__main__':
    images_dir = os.listdir("../../data/pics/")
    images_path = "../../data/pics/"

    train_path = '../../data/image_description_pair_20k_train.csv'
    test_path = '../../data/image_description_pair_20k_test.csv'

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    vocab_size, max_len, unique_words = instantiate_vocab_stats(df_train)

    word_2_indices, indices_2_word = create_dicts(unique_words)
    vgg = load_vgg16_model()
    print('VGG loaded')



    model = load_model('../../data/256_20kimages9epochs.h5')
    print('My model loaded')
    image_name = input('Please input a image name: (2572254_00.jpg,' +
                       '1818371_02.jpg, 10749590_00.jpg)\n>>>')
    img = mpimg.imread(images_path + image_name)
    image = get_encoding(vgg, image_name)
    plt.imshow(img)
    plt.show()
    print('Make Argmax prediction')
    argmax = argmax_pred_caption(image_name)
    print('Caption is: ', argmax)

    print('Make Beam prediction')
    argmax = beam_search_decoder(image, 5)
    print('Caption is: ', argmax)
