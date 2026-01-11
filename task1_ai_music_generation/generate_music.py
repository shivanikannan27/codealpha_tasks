from music21 import note, chord, stream
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import random

# Load trained model
model = load_model("model/music_model.h5")

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Sanitize notes (extra safety)
notes = [str(n) for n in notes]

pitchnames = sorted(set(notes))
note_to_int = {note: i for i, note in enumerate(pitchnames)}

# Pick a random start point
start = random.randint(0, len(notes) - 100)
pattern = [note_to_int[n] for n in notes[start:start + 100]]

prediction_output = []

# Generate 300 notes
for _ in range(300):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(pitchnames))
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]
    prediction_output.append(result)
    pattern.append(index)
    pattern = pattern[1:]

# Convert to MIDI
output_notes = []

for pattern in prediction_output:
    if '.' in pattern:
        notes_in_chord = pattern.split('.')
        chord_notes = [note.Note(int(n)) for n in notes_in_chord]
        output_notes.append(chord.Chord(chord_notes))
    else:
        output_notes.append(note.Note(pattern))

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output/generated_music.mid')

print("🎵 Music generated successfully! Check output/generated_music.mid")
