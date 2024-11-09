import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Load the saved model
model = load_model('password_generator_model.h5')

# Tokenizer setup (this should match the tokenizer used during training)
# Ensure the same tokenizer that was used during training is recreated
with open('100kCommonPWD.txt', 'r') as file:
    passwords = file.read().splitlines()

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(passwords)
total_chars = len(tokenizer.word_index) + 1  # Ensure this matches the saved model's char count
max_len = max([len(password) for password in passwords])  # Ensure max_len matches the training

# Function to generate random length passwords
def generate_password(min_length=8, max_length=12):
    password = ''
    length = random.randint(min_length, max_length)
    
    for _ in range(length):
        if len(password) == 0:
            # Generate the first character randomly
            next_char_index = np.random.randint(0, len(alphabet))  # Choose only from alphabet
            password += alphabet[next_char_index]
        else:
            # Convert the current password to a sequence
            X_password = tokenizer.texts_to_sequences([password])
            X_password = tf.keras.preprocessing.sequence.pad_sequences(X_password, maxlen=max_len, padding='pre')
            # Predict the next character
            predicted_probs = model.predict(X_password)[0]
            next_char_index = np.argmax(predicted_probs)
        
        # Get the corresponding character and append it to the password
        next_char = tokenizer.index_word.get(next_char_index, '')  # Handle cases where the index is out of range
        password += next_char
    
    return password

# Example usage: Generate a random password
#for i in range(10):
generated_password = generate_password()
print(f"Generated Password: {generated_password}")

