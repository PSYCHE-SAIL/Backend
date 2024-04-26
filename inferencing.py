import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_and_predict(model_path, new_messages):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Tokenize the new messages using the same Tokenizer instance used for training
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(new_messages)
    sequences = tokenizer.texts_to_sequences(new_messages)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Perform prediction
    predictions = model.predict(padded_sequences)

    # Convert predictions to stress levels (0-5)
    predicted_stress_levels = np.argmax(predictions, axis=1)

    return predicted_stress_levels

# Path to the saved model
model_path = 'stress_classifier_model.h5'

# New messages to predict stress levels for
new_messages = [
    "hi how are you",
    "I'm worried about my upcoming interview.",
    "Too much pressure from work deadlines.",
    "Struggling to balance studies and personal life."
]

# Perform prediction using the trained model
predicted_stress_levels = load_and_predict(model_path, new_messages)

# Display the predicted stress levels for each message
for message, stress_level in zip(new_messages, predicted_stress_levels):
    print(f"Message: {message} --> Predicted Stress Level: {stress_level}")
