from keras.models import Sequential, load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import scipy.sparse as sp
import tensorflow as tf
import nltk

nltk.download('punkt')


# Define a function to preprocess the data
def preprocess_data(data):
    X = []
    y = []
    for text, label in data:
        X.append(text)
        y.append(label)
    return X, y


# Define a function to tokenize the text
def tokenize(text):
    return text.split()


# Define a function to read the dataset
def read_dataset(file_path):
    data = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' & ')
            if len(parts) != 2:
                print(f"Ignoring line: {line}")
                count += 1
                print(count)
                continue
            text, label = parts
            data.append((text, label))

    return data


# Define a function to train the model

def train_model():
    # Read the dataset
    data = read_dataset('dataset.txt')

    # Preprocess the data
    X, y = preprocess_data(data)

    # Tokenize the text
    tokenized_data = [tokenize(text) for text in X]

    # Vectorize the text using the bag-of-words model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_data])

    # Convert the sparse matrix to Compressed Sparse Row (CSR) format
    X = sp.csr_matrix(X)

    # Apply TF-IDF transformation to the vectorized data
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)

    # Reorder the sparse matrix to avoid out-of-order indices
    X = X.tocoo()
    X = tf.SparseTensor(indices=np.vstack((X.row, X.col)).T, values=X.data, dense_shape=X.shape)
    X = tf.sparse.reorder(X)

    # Map labels to integers
    label_map = {'positive': 0, 'negative': 1}
    y = np.array([label_map[label] for label in y])

    y = np_utils.to_categorical(y, 2)

    # Define the model architecture
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=100, batch_size=32, verbose=2)

    # Save the trained model to a file
    model.save('text_classifier.h5')

    # Save the vectorizer object to a file
    with open('vectorizer.pickle', 'wb') as handle:
        pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('transformer.pickle', 'wb') as handle:
        pickle.dump(transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_model_prev():
    # Load the trained model from file
    model = load_model('text_classifier.h5')
    while True:
        # Get user input
        text = input('Enter some text, q if exit: ')
        if text == 'q':
            break
        # Tokenize the text
        tokens = tokenize(text)

        # Load the vectorizer and transformer objects from file
        with open('vectorizer.pickle', 'rb') as handle:
            vectorizer = pickle.load(handle)
        with open('transformer.pickle', 'rb') as handle:
            transformer = pickle.load(handle)

        # Vectorize and transform the text
        X = vectorizer.transform([' '.join(tokens)]).toarray()
        X = transformer.transform(X).toarray()

        # Make predictions
        predictions = model.predict(X)

        # Get the index of the predicted class with the highest probability
        pred_class = np.argmax(predictions)

        # Map the index to the corresponding label
        label_map = {0: 'positive', 1: 'negative'}
        pred_label = label_map[pred_class]

        # Print the predicted label
        print('Prediction:', pred_label)
        print('Prediction:', predictions)


# Define a function to test the model
def test_model():
    # Load the trained model from file
    model = load_model('text_classifier.h5')
    # Load the vectorizer and transformer objects from file
    with open('vectorizer.pickle', 'rb') as handle:
        vectorizer = pickle.load(handle)
    with open('transformer.pickle', 'rb') as handle:
        transformer = pickle.load(handle)
    while True:
        # Get user input
        text = input('Enter some text, q if exit: ')
        if text == 'q':
            break
        # Tokenize the text
        tokens = tokenize(text)

        # Vectorize and transform the text
        X = vectorizer.transform([' '.join(tokens)])
        X = sp.csr_matrix(X)
        X = transformer.transform(X)

        # Convert X to a dense tensor
        X = X.toarray()

        # Convert X to a Tensor
        X = tf.convert_to_tensor(X, dtype=tf.float32, name='X')

        # Make predictions
        predictions = model.predict(X)

        # Analyze the predictions
        if predictions[0][0] > predictions[0][1]:
            print('Predicted sentiment: positive')
        else:
            print('Predicted sentiment: negative')


# # Train the model and save it to a file
# train_model()

# Test the model
test_model()

# Check the dataset for errors
# data = read_dataset('dataset.txt')
