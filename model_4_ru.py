import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('banki_ru_train.csv', nrows=27000, encoding='utf-8', on_bad_lines='skip')

value = 0
if value in dataset.index:
    print(dataset.loc[value])
else:
    print("Not in index")

print(dataset.head())

#print(dataset.pop('target'))

#print(dataset.head())

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# Extract out sentences and labels
sentences = dataset['text'].tolist()
labels = dataset['target'].tolist()

# Separate out the sentences and labels into training and test sets
training_size = 20_000

training_sequences = sentences[0:training_size]
testing_sequences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices((training_sequences, training_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((testing_sequences, testing_labels))

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

VOCAB_SIZE = 3000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

model_lstm = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model_lstm.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model_lstm.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()

sample_text = ('Хорошее приложение, адекватный сотрудник')
predictions = model_lstm.predict(np.array([sample_text]))
print(predictions)

export_path_keras = "./model_lstm_ru"
model_lstm.save(export_path_keras)

model_multi_lstm = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_multi_lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model_multi_lstm.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model_multi_lstm.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# predict on a sample text without padding.

sample_text = ("""Очень грустно получилось, но результат порадовал""")
predictions = model_multi_lstm.predict(np.array([sample_text]))
print(predictions)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.show()

export_path_keras = "./model_multi_lstm_ru"
model_multi_lstm.save(export_path_keras)

e = model_multi_lstm.layers[1]
weights = e.get_weights()[0]

import io

# Write out the embedding vectors and metadata
out_v = io.open('vecs_ru.tsv', 'w', encoding='utf-8')
out_m = io.open('meta_ru.tsv', 'w', encoding='utf-8')
vocab = encoder.get_vocabulary()
for word_num in range(1, VOCAB_SIZE-10):
  word = vocab[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()