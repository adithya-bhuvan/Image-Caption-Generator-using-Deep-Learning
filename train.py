import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import define_model

# Dummy dataset (replace with Flickr8k later)
captions = {
    "img1": ["startseq a dog running endseq"],
    "img2": ["startseq a cat sitting endseq"]
}

# Tokenization
all_captions = []
for key in captions:
    all_captions.extend(captions[key])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

# Dummy image features (replace with CNN features)
features = {
    "img1": np.random.rand(4096),
    "img2": np.random.rand(4096)
}

def create_sequences(tokenizer, max_length, captions, features):
    X1, X2, y = list(), list(), list()
    for key, desc_list in captions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(features[key])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = create_sequences(tokenizer, max_length, captions, features)

# Create model
model = define_model(vocab_size, max_length)

# Train model
model.fit([X1, X2], y, epochs=10, verbose=1)

# Save model
model.save("caption_model.h5")

print("Training complete!")