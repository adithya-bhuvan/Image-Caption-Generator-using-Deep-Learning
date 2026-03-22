import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("caption_model.h5")

# Dummy tokenizer (must match training)
word_index = {'startseq':1, 'a':2, 'dog':3, 'running':4, 'endseq':5}
index_word = {v:k for k,v in word_index.items()}

max_length = 5

def generate_caption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_index.get(w, 0) for w in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Dummy image feature
photo = np.random.rand(1, 4096)

print(generate_caption(photo))