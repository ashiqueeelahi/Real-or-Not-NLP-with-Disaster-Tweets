import pandas as pd;
import tensorflow as tf;
from tensorflow.keras.preprocessing.text import Tokenizer;
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import pickle;
import keras;
from keras_preprocessing import image;
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam;
from keras.callbacks import ModelCheckpoint;
from keras.models import Sequential;
from tensorflow.keras.applications import VGG16;
from tensorflow.keras.applications import InceptionResNetV2;
from keras.applications.vgg16 import preprocess_input;
from tensorflow.keras.applications.vgg16 import decode_predictions;

#for emails/spam comments
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

import os;
from os import listdir;
from PIL import Image as PImage;
import cv2

train = pd.read_csv('../input/nlp-getting-started/train.csv');
test = pd.read_csv('../input/nlp-getting-started/test.csv');
ss = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

train.head()

x = train.text;
y = train.target;

xtrain, xvalid,ytrain, yvalid = train_test_split(x, y, test_size = 0.2, random_state= 55)

ytrain = np.array(ytrain);
yvalid = np.array(yvalid)

max_len = 30;
padding_type = 'post';
trunc_type = 'post

token = Tokenizer(num_words = 300,   oov_token = '<00V>');
token.fit_on_texts(xtrain);

word_index = token.word_index;

xtrainseq = token.texts_to_sequences(xtrain);
xtrain_pad = pad_sequences(xtrainseq, maxlen = max_len, padding = padding_type, truncating = trunc_type);

xvalidseq = token.texts_to_sequences(xvalid);
xvalid_pad = pad_sequences(xvalidseq, maxlen = max_len, padding = padding_type, truncating = trunc_type);

test = test.text;

testseq = token.texts_to_sequences(test);
test_pad = pad_sequences(testseq, maxlen = max_len, padding = padding_type, truncating = trunc_type);

model_tesnor = tf.keras.Sequential([
    tf.keras.layers.Embedding(300, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model_tesnor.compile(Adam(lr = 0.00003), loss='binary_crossentropy', metrics=['accuracy'])

model_tesnor.fit(xtrain_pad, ytrain, epochs = 13, validation_data = (xvalid_pad, yvalid), verbose = 2)

s1 = model_tesnor.predict(test_pad)

s1 = (s1 >=0.3)*1

s1

modelData = pd.DataFrame(s1, columns = ['target']);
modelData.set_index('target').to_csv('submission1.csv')

