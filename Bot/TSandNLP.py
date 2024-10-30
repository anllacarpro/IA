import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the data generated
df = pd.read_csv("data/libro.csv")

questions = df['text'].values
answers = df['response'].values

# Split data into training, validation, and test sets
train_questions, test_questions, train_answers, test_answers = train_test_split(
    questions, answers, test_size=0.2, random_state=42)

# Tokenize the questions and answers
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_questions)

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_questions)
test_sequences = tokenizer.texts_to_sequences(test_questions)

# Pad sequences to ensure uniform input size
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post')
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, padding='post')

# Encode answers similarly
answer_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
answer_tokenizer.fit_on_texts(train_answers)
train_answer_sequences = answer_tokenizer.texts_to_sequences(train_answers)
test_answer_sequences = answer_tokenizer.texts_to_sequences(test_answers)
train_answer_padded = tf.keras.preprocessing.sequence.pad_sequences(train_answer_sequences, padding='post')
test_answer_padded = tf.keras.preprocessing.sequence.pad_sequences(test_answer_sequences, padding='post')

# Define the model
model = Sequential([
    Embedding(10000, 64, input_length=train_padded.shape[1]),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dense(10000, activation='softmax')  # Choose output size based on answer vocab size
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_padded, np.array(train_answer_padded), 
    validation_data=(test_padded, np.array(test_answer_padded)), 
    epochs=10, batch_size=64
)

# Evaluate the model and print report
loss, accuracy = model.evaluate(test_padded, np.array(test_answer_padded))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Generate report using metrics
from sklearn.metrics import classification_report

preds = model.predict(test_padded)
preds_classes = np.argmax(preds, axis=1)
print(classification_report(test_answer_padded, preds_classes, target_names=answer_tokenizer.word_index.keys()))
