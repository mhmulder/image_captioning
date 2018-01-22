import numpy as np
import pandas as pd
# from pymongo import MongoClient

# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import os

from keras.preprocessing import image, sequence
from keras.applications import VGG16
from keras.models import  Model

import pickle


def preprocess_input(img):
    # convert from RGB to BGR
    # subtract means from imagenet to center around zero,
    # means are [123.68, 116.78, 103.94] RGB
    img = img[:, :, :, ::-1] #RGB to BGR
    img[:, :, :, 0] -= 103.94
    img[:, :, :, 1] -= 116.78
    img[:, :, :, 2] -= 123.68
    return img

def preprocessing(img_path):
    img = image.load_img(img_path, target_size=(224,224,3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_encoding(model, img):
    image = preprocessing(images_path+img)
    pred = model.predict(image)
    features = np.reshape(pred, pred.shape[1])
    return features

def vocab_from_df(df):
    sentences = list(df['descriptions'].values)

    words_in_each_sentence = [i.split() for i in sentences]
    unique = []
    for i in words_in_each_sentence:
        unique.extend(i)

    unique = sorted(list(set(unique)))


    return sentences

def load_vgg16_model():
    vgg16 = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
    vgg = Model(inputs=vgg16.input, outputs=vgg16.layers[-2].output)
    return vgg

def encode_images_with_model_features(df, model, pickle_filename, verbose = False):
    encoded_images = {}
    i = 0
    for img in df['image_name']:
        encoded_images[img] = get_encoding(model, img)
        i += 1
        if i % 100 == 0 and verbose == True:
            print ('Enoded: {} images.'.format(i))
    with open( pickle_filename, "wb" ) as pickle_f:
        pickle.dump(encoded_images, pickle_f )
    if verbose: print ('pickle created as {}'.format(pickle_filename))
    return encoded_images

def instantiate_vocab_stats(df):
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
    word_2_indices = {val:index for index, val in enumerate(unique_words)}
    indices_2_word = {index:val for index, val in enumerate(unique_words)}

    return word_2_indices, indices_2_word

def pad_sequences(df, max_len, vocab_size, word_2_indices):
    padded_sequences, subsequent_words = [], []

    for ix in range(df.shape[0]):
        partial_seqs = []
        next_words = []
        text = df['description'][ix].split()
        text = [word_2_indices[i] for i in text]
        for i in range(1, len(text)):
            partial_seqs.append(text[:i])
            next_words.append(text[i])
        padded_partial_seqs = sequence.pad_sequences(partial_seqs, max_len, padding='post')

        next_words_1hot = np.zeros([len(next_words), vocab_size], dtype=np.bool)

        for i,next_word in enumerate(next_words):
            next_words_1hot[i, next_word] = 1

        padded_sequences.append(padded_partial_seqs)
        subsequent_words.append(next_words_1hot)

    padded_sequences = np.asarray(padded_sequences)
    subsequent_words = np.asarray(subsequent_words)

    return padded_sequences, subsequent_words

def create_abbreviated_arrays_for_model(num_images):
    captions = []
    next_words = []

    for ix in range(num_images):#img_to_padded_seqs.shape[0]):
        captions.extend(padded_sequences[ix])
        next_words.extend(subsequent_words[ix])
        if ix % 100 == 0:
            print ('Finished: {}'.format(ix))

    captions = np.array(captions)
    next_words = np.array(next_words)
    np.save("../../data/captions8k.npy", captions)
    np.save("../../data/next_words8k.npy", next_words)


    imgs = []

    for ix in range(df_train.shape[0]):
        imgs.append(train_encoded_images[df_train['image_name'][ix]])
    imgs = np.asarray(imgs)


    images = []

    for ix in range(num_images):
        for iy in range(padded_sequences[ix].shape[0]):
            images.append(imgs[ix])
            if ix % 100 == 0:
                print ('Finished: {}'.format(ix))

    images = np.asarray(images)

    np.save("../../data/images8k.npy", images)

    print (images.shape)

    image_names = []

    for ix in range(num_images):
        for iy in range(padded_sequences[ix].shape[0]):
            image_names.append(df_train['image_name'][ix])
            if ix % 100 == 0:
                print ('Finished: {}'.format(ix))

    image_names = np.asarray(image_names)

    np.save("../../data/image_names8k.npy", image_names)

    print (len(image_names))



if __name__ == '__main__':
    images_dir = os.listdir("../../data/pics/")
    images_path = "../../data/pics/"

    train_path='../../data/image_description_pair_8k_train.csv'
    test_path='../../data/image_description_pair_8k_test.csv'

    # read_csv
    df_train = pd.read_csv(train_path)[:100]
    df_test = pd.read_csv(test_path)[:100]
    print ("CSV's loaded!")
    ## image preprocessing
    # load keras VGG16 model
    vgg = load_vgg16_model()
    print ("VGG16 Model loaded!")
    train_encoded_images = encode_images_with_model_features(df_train, vgg, "../../data/train_encoded_images.pkl", verbose =True)
    print ('All train images encoded!')
    test_encoded_images = encode_images_with_model_features(df_test, vgg, "../../data/test_encoded_images.pkl", verbose=True)
    print ('All test images encoded!')
    ## language preprocessing
    vocab_size, max_len, unique_words = instantiate_vocab_stats(df_train)

    word_2_indices, indices_2_word = create_dicts(unique_words)

    padded_sequences, subsequent_words = pad_sequences(df_train, max_len, vocab_size, word_2_indices)

    print ('All sequences padded')



    # number of images to train on
    number_of_images = 100

    create_abbreviated_arrays_for_model(number_of_images)

    print ('Finished')
