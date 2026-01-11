from music21 import converter, instrument, note, chord
import os
import pickle

notes = []

for file in os.listdir("dataset/midi_songs"):
    if file.endswith(".mid"):
        midi = converter.parse(f"dataset/midi_songs/{file}")
        parts = instrument.partitionByInstrument(midi)

        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flat.notes

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("✅ Preprocessing completed correctly")
