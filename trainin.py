import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

# Sample data
# Load data from JSON file
def load_data_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Path to the JSON file
json_file_path = 'data.json'

# Load data
data = load_data_from_json(json_file_path)

messages = [entry['message'] for entry in data]
stress_levels = [entry['stress_level'] for entry in data]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)
sequences = tokenizer.texts_to_sequences(messages)
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  # 6 classes for stress levels 0-5
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert stress levels to numpy array
stress_levels = np.array(stress_levels)

# Train the model
model.fit(padded_sequences, stress_levels, epochs=10)

# Save the model
model.save('stress_classifier_model.h5')
