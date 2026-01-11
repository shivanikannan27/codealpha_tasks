import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# 🔒 FORCE-SANITIZE NOTES (CRITICAL FIX)
clean_notes = []
for n in notes:
    if isinstance(n, list):
        clean_notes.append(".".join(str(x) for x in n))
    else:
        clean_notes.append(str(n))

notes = clean_notes

print("Sample notes:", notes[:10])
print("Type check:", type(notes[0]))

sequence_length = 100

# Create vocabulary safely
pitchnames = sorted(set(map(str, notes)))
note_to_int = {note: i for i, note in enumerate(pitchnames)}

# Prepare sequences
network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

network_input = np.array(network_input)
network_input = np.reshape(network_input, (network_input.shape[0], sequence_length, 1))
network_input = network_input / float(len(pitchnames))

network_output = to_categorical(network_output, num_classes=len(pitchnames))

# Build model
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256, activation="relu"))
model.add(Dense(len(pitchnames), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

checkpoint = ModelCheckpoint(
    "model/music_model.h5",
    monitor="loss",
    save_best_only=True
)

model.fit(
    network_input,
    network_output,
    epochs=20,
    batch_size=64,
    callbacks=[checkpoint]
)

print("✅ Training completed successfully")



