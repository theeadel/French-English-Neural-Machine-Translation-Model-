# Machine Translation with Seq2Seq and Attention

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, LSTM, Dense, RepeatVector, TimeDistributed, 
                                     Bidirectional, Input, Attention, Concatenate)
from tensorflow.keras.optimizers import Adam
import pickle
from google.colab import files

!git clone https://github.com/zaka-ai/machine_learning_certification.git

# Load datasets
English = pd.read_csv("en.csv")
French = pd.read_csv("fr.csv")
df = pd.concat([English, French], axis=1)
df.columns = ['English', 'French']

# Clean punctuation
df['English'] = df['English'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['French'] = df['French'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Sentence lengths
df['ENG Length'] = df['English'].apply(lambda x: len(x.split()))
df['FR Length'] = df['French'].apply(lambda x: len(x.split()))

# Visualize
sns.histplot(df['FR Length'], kde=True, bins=30)
plt.title('French Sentence Lengths')
plt.show()

sns.histplot(df['ENG Length'], kde=True, bins=30)
plt.title('English Sentence Lengths')
plt.show()

max_length = max(df['ENG Length'].max(), df['FR Length'].max())

# Tokenization
tokenizer_en = Tokenizer()
tokenizer_en.fit_on_texts(df['English'])
en_sequences = tokenizer_en.texts_to_sequences(df['English'])

tokenizer_fr = Tokenizer()
tokenizer_fr.fit_on_texts(df['French'])
fr_sequences = tokenizer_fr.texts_to_sequences(df['French'])

num_words_en = len(tokenizer_en.word_index) + 1
num_words_fr = len(tokenizer_fr.word_index) + 1

# Padding
en_padded_sequences = pad_sequences(en_sequences, maxlen=max_length, padding='post')
fr_padded_sequences = pad_sequences(fr_sequences, maxlen=max_length, padding='post')

# Attention-based Encoder-Decoder
encoder_input = Input(shape=(max_length,))
encoder_embedding = Embedding(num_words_en, 256, input_length=max_length)(encoder_input)
encoder_lstm1 = Bidirectional(LSTM(256, return_sequences=True))(encoder_embedding)
encoder_lstm2 = Bidirectional(LSTM(256, return_sequences=True))(encoder_lstm1)
encoder_projection = Dense(256)(encoder_lstm2)

decoder_input = Input(shape=(max_length,))
decoder_embedding = Embedding(num_words_fr, 256, input_length=max_length)(decoder_input)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding)

attention = Attention()([decoder_outputs, encoder_projection])
context_vector = Concatenate()([decoder_outputs, attention])
output = TimeDistributed(Dense(num_words_fr, activation="softmax"))(context_vector)

model = Model(inputs=[encoder_input, decoder_input], outputs=output)
model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Decoder input shift
decoder_input_sequences = np.roll(fr_padded_sequences, shift=1, axis=1)
decoder_input_sequences[:, 0] = 0

# Train model
model.fit([
    en_padded_sequences, decoder_input_sequences],
    fr_padded_sequences,
    batch_size=64,
    epochs=10,
    validation_split=0.2
)

# Translate function
def translate_sentence(input_sentence):
    input_sequence = tokenizer_en.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding="post")
    decoder_input_sequence = np.zeros((1, max_length))
    decoder_input_sequence[0, 0] = 0

    for i in range(1, max_length):
        prediction = model.predict([input_sequence, decoder_input_sequence])
        predicted_word_id = np.argmax(prediction[0, i - 1, :])
        if predicted_word_id == 0:
            break
        decoder_input_sequence[0, i] = predicted_word_id

    translated_sentence = tokenizer_fr.sequences_to_texts(decoder_input_sequence)[0]
    return translated_sentence

# Test
print(translate_sentence("she is driving the truck"))

# Save artifacts
with open("eng_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer_en, f)

with open("fr_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer_fr, f)

model.save_weights("model.weights.h5")

files.download("eng_tokenizer.pkl")
files.download("fr_tokenizer.pkl")
files.download("model.weights.h5")
