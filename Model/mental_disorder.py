from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import nltk
import io
import json
#from google.colab import files
nltk.download('stopwords')
nltk.download('punkt')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') > 0.87):
            print("\n Target telah tercapai")
            self.model.stop_training = True

callback = myCallback()

path = "final_das.csv"

new_dataset = pd.read_csv(path)
new_dataset

# Drop missing values, duplicates, and unnecessary columns
empty_rows = new_dataset[new_dataset['text'].str.strip() == '']
new_dataset.drop(empty_rows.index)
new_dataset.dropna(inplace=True)
new_dataset.drop_duplicates(inplace=True)

new_dataset['text'] = new_dataset['text'].str.lower()

# Remove '/r/', other punctuation, hyperlinks, and hashtags
new_dataset['text'] = new_dataset['text'].str.replace(r'/r/|[^\w\s]|https?://\S+|www\.\S+|\#\w+', '', regex=True)

# Remove stopwords
stop_words = set(stopwords.words('indonesian'))
new_dataset['text'] = new_dataset['text'].apply(lambda text: ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words]))
new_dataset.sort_values(by="target", ascending=True, inplace=True,ignore_index = True)
new_dataset = new_dataset[new_dataset['text'] != '']
new_dataset['target'].astype(int)
print(new_dataset)

from sklearn.model_selection import train_test_split


# Defining features and labels
X = new_dataset['text'].values
y = new_dataset['target'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from keras.utils import to_categorical

# Assuming y_train is an array of integer class labels
y_train_encoded = to_categorical(y_train, num_classes=4)


X

# Info
vocab_size = 10000
embedding_dim = 200
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Tokenization training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(X_train)
training_padded_sequences = pad_sequences(training_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded_sequences = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer_reddit.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00065), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(training_padded_sequences, y_train, callbacks=callback, epochs=200, validation_data=(test_padded_sequences, y_test))
model.save("model_reddit.h5")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
