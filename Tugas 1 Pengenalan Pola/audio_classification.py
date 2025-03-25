# -*- coding: utf-8 -*-
"""""
Import necessary libraries for audio classification and visualization.

This block imports various Python libraries and modules that are essential for
audio data processing, machine learning model creation, and data visualization.
Imported libraries:
    - os: For operating system dependent functionality
    - matplotlib.pyplot: For creating static, animated, and interactive visualizations
    - numpy: For numerical computing with Python
    - seaborn: For statistical data visualization
    - tensorflow: For machine learning and neural network operations

Imported TensorFlow modules:
    - layers, Sequential: For building neural network models
    - EarlyStopping: A callback for stopping training when a monitored metric has stopped improving

Other imports:
    - display from IPython: For displaying outputs in Jupyter notebooks
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from IPython import display

"""Set the random seed for reproducibility.

This block sets the random seed for TensorFlow and NumPy to ensure reproducibility of results.
By fixing the seed, the random number generators will produce the same sequence of numbers
each time the code is run, which is useful for debugging and consistent experimentation.

Defined variables:
    - SEED: An integer value (42) used as the seed for random number generation.

Operations:
    - tf.random.set_seed(SEED): Sets the random seed for TensorFlow.
    - np.random.seed(SEED): Sets the random seed for NumPy.
"""

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

"""Define constants for spectrogram generation.

This block defines two constants used for generating spectrograms from audio waveforms:
- `FRAME_LENGTH`: The length of each frame in the Short-Time Fourier Transform (STFT). It determines the number of samples in each frame.
- `FRAME_STEP`: The step size or stride between consecutive frames in the STFT. It determines the overlap between frames.

Defined variables:
- `FRAME_LENGTH`: An integer value (255) representing the frame length.
- `FRAME_STEP`: An integer value (128) representing the frame step.

These constants are used in the `get_spectrogram` method of the `Spectrogram` class to compute the spectrogram of an audio waveform.
"""

FRAME_LENGTH = 255
FRAME_STEP = 128

"""Define the `Spectrogram` class for spectrogram generation and visualization.

This block defines a class `Spectrogram` that provides methods to generate and visualize spectrograms from audio waveforms. The class includes the following methods:

### Methods:
1. **`__init__`**:
    - Initializes the `Spectrogram` class.
    - No parameters or attributes are defined in the constructor.

2. **`get_spectrogram(waveform)`**:
    - Computes the spectrogram of an audio waveform using the Short-Time Fourier Transform (STFT).
    - Parameters:
      - `waveform` (tf.Tensor): A 1D tensor representing the audio waveform.
    - Returns:
      - `spectrogram` (tf.Tensor): A 3D tensor representing the magnitude of the STFT with an additional channel dimension.
    - Operations:
      - Uses `tf.signal.stft` to compute the STFT of the waveform with the constants `FRAME_LENGTH` and `FRAME_STEP`.
      - Takes the absolute value of the STFT to get the magnitude.
      - Adds a new axis to the spectrogram for compatibility with machine learning models.

3. **`plot_spectrogram(spectrogram, ax)`**:
    - Visualizes the spectrogram using a logarithmic scale.
    - Parameters:
      - `spectrogram` (np.ndarray or tf.Tensor): A 2D or 3D tensor representing the spectrogram.
      - `ax` (matplotlib.axes._axes.Axes): A matplotlib axis object where the spectrogram will be plotted.
    - Operations:
      - If the spectrogram has more than 2 dimensions, it squeezes the last axis to make it 2D.
      - Computes the logarithm of the spectrogram to enhance visualization.
      - Uses `ax.pcolormesh` to plot the spectrogram on the provided axis.
"""

class Spectrogram():
    def __init__(self):
        pass

    def get_spectrogram(self, waveform):
        spectrogram = tf.signal.stft(waveform,
                                     frame_length=FRAME_LENGTH,
                                     frame_step=FRAME_STEP)
        spectrogram = tf. abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def plot_spectrogram(self, spectrogram, ax):
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)

        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

"""Define constants for dataset configuration.

This block defines constants used for configuring the dataset loading process. These constants include the batch size, output sequence length, and paths to the training and validation datasets.

### Defined Variables:
1. **`BATCH_SIZE`**:
    - Type: `int`
    - Value: `32`
    - Description: The number of samples per batch during training and validation.

2. **`OUTPUT_SEQUENCE_LENGTH`**:
    - Type: `int`
    - Value: `16000`
    - Description: The desired length of the audio sequences. Audio samples will be padded or truncated to this length.

3. **`TRAIN_DATASET_PATH`**:
    - Type: `str`
    - Value: `'dataset/cats_dogs/train'`
    - Description: The file path to the training dataset directory. This directory contains subdirectories for each class.

4. **`VALIDATION_DATASET_PATH`**:
    - Type: `str`
    - Value: `'dataset/cats_dogs/test'`
    - Description: The file path to the validation dataset directory. This directory contains subdirectories for each class.

These constants are used in the `AudioDataLoader` class to load and preprocess the audio datasets.
"""

BATCH_SIZE = 32
OUTPUT_SEQUENCE_LENGTH = 16000
TRAIN_DATASET_PATH = 'dataset/cats_dogs/train'
VALIDATION_DATASET_PATH = 'dataset/cats_dogs/test'

"""Define the `AudioDataLoader` class for loading and preprocessing audio datasets.

This block defines a class `AudioDataLoader` that provides methods to load and preprocess audio datasets for training, validation, and testing. The class includes the following methods:

### Methods:
1. **`__init__`**:
    - Initializes the `AudioDataLoader` class.
    - No parameters or attributes are defined in the constructor.

2. **`get_training_dataset_and_class_names(self)`**:
    - Loads the training dataset from the specified directory and retrieves the class names.
    - Returns:
      - `train_ds` (tf.data.Dataset): A dataset containing audio waveforms and their corresponding labels.
      - `class_names` (list): A list of class names corresponding to the dataset.

3. **`get_validation_and_test_dataset(self)`**:
    - Loads the validation dataset from the specified directory and splits it into validation and test datasets.
    - Returns:
      - `validation_ds` (tf.data.Dataset): A dataset containing validation audio waveforms and their corresponding labels.
      - `test_ds` (tf.data.Dataset): A dataset containing test audio waveforms and their corresponding labels.

4. **`_get_test_dataset(self, validation_dataset)`**:
    - Splits the validation dataset into two equal parts to create validation and test datasets.
    - Parameters:
      - `validation_dataset` (tf.data.Dataset): The original validation dataset.
    - Returns:
      - `validation_dataset` (tf.data.Dataset): The first half of the original dataset.
      - `test_dataset` (tf.data.Dataset): The second half of the original dataset.

5. **`_squeeze(self, waveforms, labels)`**:
    - Removes the last axis from the audio waveforms to simplify their shape.
    - Parameters:
      - `waveforms` (tf.Tensor): A tensor containing audio waveforms.
      - `labels` (tf.Tensor): A tensor containing the corresponding labels.
    - Returns:
      - `waveforms` (tf.Tensor): A tensor with the last axis removed.
      - `labels` (tf.Tensor): The unchanged labels tensor.

### Constants Used:
- `TRAIN_DATASET_PATH`: Path to the training dataset directory.
- `VALIDATION_DATASET_PATH`: Path to the validation dataset directory.
- `BATCH_SIZE`: Number of samples per batch.
- `OUTPUT_SEQUENCE_LENGTH`: Desired length of audio sequences.
- `SEED`: Random seed for reproducibility.
"""

class AudioDataLoader():
    def __init__(self):
        pass

    def get_training_dataset_and_class_names(self):
        train_ds = tf.keras.utils.audio_dataset_from_directory(
            TRAIN_DATASET_PATH,
            batch_size=BATCH_SIZE,
            output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
            seed=SEED
        )

        class_names = train_ds.class_names
        train_ds = train_ds.map(self._squeeze, tf.data.AUTOTUNE)
        return train_ds, class_names

    def get_validation_and_test_dataset(self):
        validation_ds = tf.keras.utils.audio_dataset_from_directory(
            VALIDATION_DATASET_PATH,
            batch_size=BATCH_SIZE,
            output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
            seed=SEED
        )

        validation_ds = validation_ds.map(self._squeeze, tf.data.AUTOTUNE)
        validation_ds, test_ds = self._get_test_dataset(validation_ds)
        return validation_ds, test_ds

    def _get_test_dataset(self, validation_dataset):
        validation_dataset = validation_dataset.shard(num_shards=2, index=0)
        test_dataset = validation_dataset.shard(num_shards=2, index=1)
        return validation_dataset, test_dataset

    def _squeeze(self, waveforms, labels):
        waveforms = tf.squeeze(waveforms, axis=-1)
        return waveforms, labels

"""Count the number of files in the dataset directories.

This block defines a function `count_files_in_dataset` to count the number of files in each class directory within a dataset path. It also calculates the total number of files across all classes. The function is used to analyze both the training and validation datasets.

### Function:
1. **`count_files_in_dataset(dataset_path)`**:
    - Counts the number of files in each class directory within the specified dataset path.
    - Parameters:
      - `dataset_path` (str): The path to the dataset directory containing subdirectories for each class.
    - Returns:
      - `class_counts` (dict): A dictionary where keys are class names and values are the number of files in each class.
      - `total_count` (int): The total number of files across all classes.
    - Operations:
      - Iterates through the subdirectories in the dataset path.
      - Checks if each subdirectory is a valid directory.
      - Counts the number of files in each subdirectory and updates the total count.

### Outputs:
- Prints the number of samples for each class and the total number of samples in the training and validation datasets.

### Example Output:

"""

def count_files_in_dataset(dataset_path):
    class_counts = {}
    total_count = 0

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            class_count = len(os.listdir(class_dir))
            class_counts[class_name] = class_count
            total_count += class_count

    return class_counts, total_count

train_ds_counts, train_ds_total = count_files_in_dataset(TRAIN_DATASET_PATH)
validation_ds_counts, validation_ds_total = count_files_in_dataset(VALIDATION_DATASET_PATH)

print('Train Dataset:')
for class_name, count in train_ds_counts.items():
    print(f'{class_name}: {count} samples')
print(f'Total samples: {train_ds_total}')

print('\nValidation Dataset:')
for class_name, count in validation_ds_counts.items():
    print(f'{class_name}: {count} samples')
print(f'Total samples: {validation_ds_total}')

"""Load and preprocess the audio datasets.

This block initializes an instance of the `AudioDataLoader` class and uses its methods to load and preprocess the training, validation, and test datasets. The datasets are loaded from the specified directory paths and are prepared for further processing.

### Operations:
1. **Initialize `AudioDataLoader`**:
    - Creates an instance of the `AudioDataLoader` class to handle dataset loading and preprocessing.

2. **Load Training Dataset**:
    - Calls the `get_training_dataset_and_class_names` method to load the training dataset and retrieve the class names.
    - Returns:
      - `train_ds` (tf.data.Dataset): A dataset containing audio waveforms and their corresponding labels for training.
      - `class_names` (list): A list of class names corresponding to the dataset.

3. **Load Validation and Test Datasets**:
    - Calls the `get_validation_and_test_dataset` method to load the validation dataset and split it into validation and test datasets.
    - Returns:
      - `validation_ds` (tf.data.Dataset): A dataset containing audio waveforms and their corresponding labels for validation.
      - `test_ds` (tf.data.Dataset): A dataset containing audio waveforms and their corresponding labels for testing.

### Constants Used:
- `TRAIN_DATASET_PATH`: Path to the training dataset directory.
- `VALIDATION_DATASET_PATH`: Path to the validation dataset directory.
- `BATCH_SIZE`: Number of samples per batch.
- `OUTPUT_SEQUENCE_LENGTH`: Desired length of audio sequences.
- `SEED`: Random seed for reproducibility.

### Outputs:
- `train_ds`: Training dataset.
- `validation_ds`: Validation dataset.
- `test_ds`: Test dataset.
- `class_names`: List of class names.
"""

dataloader = AudioDataLoader()
train_ds, class_names = dataloader.get_training_dataset_and_class_names()
validation_ds, test_ds = dataloader.get_validation_and_test_dataset()

"""Print class names and inspect the shape of waveforms and labels in the training dataset.

This block of code performs the following operations:

### Operations:
1. **Print Class Names**:
    - Outputs the list of class names (`class_names`) to the console.
    - `class_names` is a list of strings representing the labels of the audio dataset (e.g., `['cat', 'dog']`).

2. **Inspect Training Dataset**:
    - Iterates through the first batch of the training dataset (`train_ds`) using the `.take(1)` method.
    - Prints the shape of the waveforms and labels in the batch:
        - `waveforms.shape`: A tensor representing the shape of the audio waveforms in the batch. The shape is `(BATCH_SIZE, OUTPUT_SEQUENCE_LENGTH)`, where `BATCH_SIZE` is the number of samples in the batch, and `OUTPUT_SEQUENCE_LENGTH` is the length of each audio sequence.
        - `labels.shape`: A tensor representing the shape of the labels in the batch. The shape is `(BATCH_SIZE,)`, where `BATCH_SIZE` is the number of samples in the batch.
"""

print(class_names)
for waveforms, labels in train_ds.take(1):
    print(f'Waveform shape: {waveforms.shape}')
    print(f'Labels shape: {labels.shape}')

"""Visualize waveforms of audio samples in the training dataset.

This block of code generates a grid of subplots to visualize the waveforms of the first 9 audio samples in the training dataset. Each subplot corresponds to a single audio sample and displays the waveform along with its class label.

### Operations:
1. **Create a Figure**:
    - Initializes a figure with a size of `(16, 9)` for the grid of subplots.
    - Adjusts the bottom margin of the figure to ensure proper spacing between subplots.

2. **Iterate Over Samples**:
    - Loops through the first 9 audio samples in the `waveforms` tensor.
    - For each sample:
        - Creates a subplot in a 3x3 grid.
        - Plots the waveform using the `plt.plot()` function.
        - Sets the title of the subplot to the corresponding class label from the `class_names` list.
        - Configures the axis limits to display the waveform within the range `[0, 16000]` for the x-axis and `[-1, 1]` for the y-axis.

### Inputs:
- `waveforms`: A tensor of shape `(32, 16000)` containing audio waveforms for the current batch.
- `labels`: A tensor of shape `(32,)` containing the class labels for the current batch.
- `class_names`: A list of strings representing the class names (e.g., `['cat', 'dog']`).

### Outputs:
- A grid of 9 subplots displaying the waveforms of the first 9 audio samples in the batch, with each subplot labeled by its corresponding class name.

### Example Visualization:
- The first subplot might display a waveform labeled as "cat".
- The second subplot might display a waveform labeled as "dog".
- The waveforms are plotted with consistent axis limits for better comparison.
"""

plt.figure(figsize=(16, 9))
plt.subplots_adjust(bottom=0.2)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.plot(waveforms[i].numpy())
    plt.title(class_names[labels[i]])
    plt.axis([0, 16000, -1, 1])

spec = Spectrogram()
for j in range(6):
    label = class_names[labels[j]]
    waveform = waveforms[j]
    spectrogram = spec.get_spectrogram(waveform)
    print(f'\nLabel: {label}')
    print(f'Spectrogram Shape: {spectrogram.shape}')
    display.display(display.Audio(waveform, rate=16000))

fig, axes = plt.subplots(2, figsize=(12, 8))
plt.subplots_adjust(bottom=0.2)

timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

spec.plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()

"""### Map Audio Waveforms to Spectrograms

This block of code maps audio waveforms to their corresponding spectrograms for the training, validation, and test datasets. The spectrograms are generated using the `get_spectrogram` method of the `Spectrogram` class. The resulting datasets contain spectrograms and their associated labels, which are used as inputs for the machine learning model.

### Operations:
1. **Map Training Dataset**:
    - Applies the `get_spectrogram` method to each audio waveform in the training dataset (`train_ds`).
    - The `map_func` lambda function takes an audio waveform and its label as input and returns the spectrogram and label as output.
    - The mapping operation is parallelized using `tf.data.AUTOTUNE` for optimized performance.

2. **Map Validation Dataset**:
    - Similar to the training dataset, the `get_spectrogram` method is applied to each audio waveform in the validation dataset (`validation_ds`).
    - The resulting dataset contains spectrograms and their corresponding labels.

3. **Map Test Dataset**:
    - The `get_spectrogram` method is applied to each audio waveform in the test dataset (`test_ds`).
    - The resulting dataset contains spectrograms and their corresponding labels.

### Inputs:
- `train_ds`: Training dataset containing audio waveforms and their labels.
- `validation_ds`: Validation dataset containing audio waveforms and their labels.
- `test_ds`: Test dataset containing audio waveforms and their labels.
- `spec`: An instance of the `Spectrogram` class used to generate spectrograms.

### Outputs:
- `train_spectrogram_ds`: Training dataset containing spectrograms and their labels.
- `validation_spectrogram_ds`: Validation dataset containing spectrograms and their labels.
- `test_spectrogram_ds`: Test dataset containing spectrograms and their labels.

### Example Usage:
- The spectrogram datasets (`train_spectrogram_ds`, `validation_spectrogram_ds`, `test_spectrogram_ds`) are used as inputs for training, validating, and testing the machine learning model.
- These datasets are further optimized using caching and prefetching in subsequent steps.
"""

train_spectrogram_ds = train_ds.map(
    map_func=lambda audio, label:(spec.get_spectrogram(audio), label),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_spectrogram_ds = validation_ds.map(
    map_func=lambda audio, label:(spec.get_spectrogram(audio), label),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_spectrogram_ds = test_ds.map(
    map_func=lambda audio, label:(spec.get_spectrogram(audio), label),
    num_parallel_calls=tf.data.AUTOTUNE
)

"""### Extract a Batch of Spectrograms and Labels

This block of code extracts a single batch of spectrograms and their corresponding labels from the `train_spectrogram_ds` dataset. The dataset is iterated using the `.take(1)` method, which retrieves the first batch. The `break` statement ensures that only one batch is extracted.

### Operations:
1. **Iterate Over Dataset**:
    - The `for` loop iterates over the `train_spectrogram_ds` dataset.
    - The `.take(1)` method limits the iteration to the first batch.

2. **Extract Batch**:
    - The `example_spectrograms` variable stores the spectrograms in the batch.
    - The `example_labels` variable stores the corresponding labels for the spectrograms.

3. **Break Loop**:
    - The `break` statement exits the loop after extracting the first batch.

### Inputs:
- `train_spectrogram_ds`: A preprocessed dataset containing spectrograms and their corresponding labels.

### Outputs:
- `example_spectrograms`: A tensor of shape `(BATCH_SIZE, 124, 129, 1)` containing spectrograms for the batch.
- `example_labels`: A tensor of shape `(BATCH_SIZE,)` containing the labels for the spectrograms in the batch.

### Example Usage:
- This block is useful for inspecting a single batch of data, visualizing spectrograms, or debugging the dataset pipeline.
"""

for example_spectrograms, example_labels in train_spectrogram_ds.take(1):
    break

"""### Visualize Spectrograms for a Batch of Audio Samples

This block of code visualizes the spectrograms for a batch of audio samples in a grid layout. Each spectrogram is displayed in a subplot, and the corresponding class label is shown as the title of the subplot.

### Operations:
1. **Define Grid Dimensions**:
    - `rows`: Number of rows in the grid (3).
    - `cols`: Number of columns in the grid (3).
    - `n`: Total number of spectrograms to display (`rows * cols`).

2. **Create Subplots**:
    - Initializes a figure with a grid of subplots using `plt.subplots`.
    - Sets the figure size to `(16, 9)` for better visualization.
    - Adjusts the bottom margin of the figure to ensure proper spacing between subplots.

3. **Iterate Over Spectrograms**:
    - Loops through the first `n` spectrograms in the `example_spectrograms` tensor.
    - For each spectrogram:
        - Determines the row (`r`) and column (`c`) indices for the subplot.
        - Retrieves the corresponding axis (`ax`) from the `axes` array.
        - Calls the `plot_spectrogram` method of the `Spectrogram` class to visualize the spectrogram on the axis.
        - Sets the title of the subplot to the corresponding class label from the `class_names` list.

4. **Display the Plot**:
    - Renders the figure with the grid of spectrograms.

### Inputs:
- `example_spectrograms`: A tensor of shape `(32, 124, 129, 1)` containing spectrograms for the current batch.
- `example_labels`: A tensor of shape `(32,)` containing the class labels for the spectrograms in the batch.
- `class_names`: A list of strings representing the class names (e.g., `['cat', 'dog']`).
- `spec`: An instance of the `Spectrogram` class used to generate and visualize spectrograms.

### Outputs:
- A grid of 9 subplots displaying the spectrograms of the first 9 audio samples in the batch, with each subplot labeled by its corresponding class name.

### Example Visualization:
- The first subplot might display a spectrogram labeled as "dog".
- The second subplot might display a spectrogram labeled as "cat".
- The spectrograms are plotted with consistent axis limits for better comparison.
"""

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
plt.subplots_adjust(bottom=0.2)

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    spec.plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(class_names[example_labels[i]])

plt.show()

"""### Cache and Prefetch Datasets

This block of code optimizes the training, validation, and test datasets by applying caching and prefetching. These operations improve the performance of the data pipeline by reducing the latency during data loading and preprocessing.

### Operations:
1. **Cache Datasets**:
    - The `.cache()` method caches the dataset in memory or on disk after the first iteration. This avoids redundant computations and speeds up subsequent iterations.

2. **Prefetch Datasets**:
    - The `.prefetch(tf.data.AUTOTUNE)` method overlaps the data preprocessing and model execution. It allows the data pipeline to fetch the next batch of data while the current batch is being processed by the model.

### Inputs:
- `train_spectrogram_ds`: Training dataset containing spectrograms and their labels.
- `validation_spectrogram_ds`: Validation dataset containing spectrograms and their labels.
- `test_spectrogram_ds`: Test dataset containing spectrograms and their labels.

### Outputs:
- Optimized datasets (`train_spectrogram_ds`, `validation_spectrogram_ds`, `test_spectrogram_ds`) with caching and prefetching applied.

### Example Usage:
- These optimized datasets are used as inputs for training, validating, and testing the machine learning model. The caching and prefetching operations ensure efficient data loading and preprocessing during model training.
"""

train_spectrogram_ds = train_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
validation_spectrogram_ds = validation_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print(input_shape
      )
num_labels = len(class_names)
normalization = layers.Normalization()
normalization.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    normalization,
    layers.Conv2D(32,3, activation='relu'),
    layers.Dropout(0.2),
    layers.Conv2D(128,3, activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_labels)
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

"""### Train the Model

This block of code trains the convolutional neural network (CNN) model on the training dataset (`train_spectrogram_ds`) and evaluates it on the validation dataset (`validation_spectrogram_ds`). The training process is configured to run for a maximum of 30 epochs, with early stopping enabled to halt training if the validation loss does not improve for 5 consecutive epochs.

### Operations:
1. **Train the Model**:
    - The `model.fit()` method is used to train the model.
    - **Inputs**:
        - `train_spectrogram_ds`: The training dataset containing spectrograms and their corresponding labels.
        - `validation_data`: The validation dataset used to evaluate the model's performance after each epoch.
        - `epochs`: The maximum number of epochs for training (30 in this case).
        - `callbacks`: A list of callbacks to apply during training. The `earlystop` callback is included to monitor the validation loss and stop training early if necessary.

2. **Store Training History**:
    - The `history` object returned by `model.fit()` contains the training and validation metrics (e.g., loss and accuracy) for each epoch.
    - This information is used for visualizing the training progress and evaluating the model's performance.

### Inputs:
- `train_spectrogram_ds`: A preprocessed dataset containing spectrograms and their labels for training.
- `validation_spectrogram_ds`: A preprocessed dataset containing spectrograms and their labels for validation.
- `earlystop`: An instance of the `EarlyStopping` callback to monitor validation loss.
- `epochs`: The maximum number of training epochs (30).

### Outputs:
- `history`: A `History` object containing the training and validation metrics for each epoch.

### Example Usage:
- The `history` object can be used to plot the training and validation loss and accuracy over epochs to analyze the model's performance.
- The trained model is evaluated on the test dataset (`test_spectrogram_ds`) in subsequent steps.
"""

history = model.fit(
    train_spectrogram_ds,
    validation_data=validation_spectrogram_ds,
    epochs=30,
    callbacks=[earlystop]
)

"""### Visualize Training and Validation Metrics

This block of code visualizes the training and validation loss and accuracy over epochs. It uses the `history` object returned by the `model.fit()` method to extract the metrics and plots them in two subplots: one for loss and the other for accuracy.

### Operations:
1. **Extract Metrics**:
    - Retrieves the training and validation metrics from the `history.history` dictionary.
    - The dictionary contains the following keys:
        - `'loss'`: Training loss for each epoch.
        - `'val_loss'`: Validation loss for each epoch.
        - `'accuracy'`: Training accuracy for each epoch.
        - `'val_accuracy'`: Validation accuracy for each epoch.

2. **Create Figure**:
    - Initializes a figure with a size of `(16, 6)` for the subplots.

3. **Plot Loss**:
    - Creates the first subplot to visualize the training and validation loss.
    - Plots the loss values against the epochs for both training and validation.
    - Adds a legend to distinguish between training and validation loss.
    - Sets the y-axis limit to start from 0 and adjusts the upper limit dynamically.

4. **Plot Accuracy**:
    - Creates the second subplot to visualize the training and validation accuracy.
    - Plots the accuracy values against the epochs for both training and validation.
    - Adds a legend to distinguish between training and validation accuracy.
    - Sets the y-axis limit to the range `[0, 1]`.

5. **Display the Plot**:
    - Renders the figure with the two subplots.

### Inputs:
- `history`: A `History` object containing the training and validation metrics for each epoch.
- `metrics`: A dictionary extracted from `history.history` containing the following keys:
    - `'loss'`: Training loss.
    - `'val_loss'`: Validation loss.
    - `'accuracy'`: Training accuracy.
    - `'val_accuracy'`: Validation accuracy.

### Outputs:
- A figure with two subplots:
    - The first subplot displays the training and validation loss over epochs.
    - The second subplot displays the training and validation accuracy over epochs.

### Example Visualization:
- The loss plot shows how the training and validation loss decrease over epochs, indicating model improvement.
- The accuracy plot shows how the training and validation accuracy increase over epochs, reflecting better classification performance.
"""

metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['training', 'validation'])
plt.ylim([0, max(plt.ylim())])
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['training', 'validation'])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()

model.evaluate(test_spectrogram_ds, return_dict=True)

y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat([y for x, y in test_spectrogram_ds], axis=0)

conf_matrix = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 6))

sns.heatmap(conf_matrix,
            xticklabels=class_names,
            yticklabels=class_names,
            annot=True,
            fmt='g',
            cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

"""### Predict and Visualize Audio Classification for a Single Audio File

This block of code performs the following operations:

### Operations:
1. **Load Audio File**:
    - Reads an audio file from the specified path (`x`) using TensorFlow's `tf.io.read_file` function.
    - Decodes the audio file into a waveform and its sample rate using `tf.audio.decode_wav`.
    - The waveform is squeezed to remove the last axis, resulting in a 1D tensor.

2. **Generate Spectrogram**:
    - Converts the waveform into a spectrogram using the `get_spectrogram` method of the `Spectrogram` class.
    - Adds a new axis to the spectrogram to match the input shape expected by the model.

3. **Make Prediction**:
    - Passes the spectrogram through the trained model to obtain predictions.
    - Applies the softmax function to the model's output logits to calculate the probabilities for each class.

4. **Visualize Prediction**:
    - Creates a bar plot to visualize the predicted probabilities for each class.
    - The x-axis represents the class labels (`x_labels`), and the y-axis represents the probabilities.

5. **Play Audio**:
    - Plays the original audio waveform using the `display.Audio` function from IPython.

### Inputs:
- `x`: A string representing the file path to the audio file (e.g., `'dataset/cats_dogs/test/test/dog_barking_44.wav'`).
- `spec`: An instance of the `Spectrogram` class used to generate the spectrogram.
- `model`: A trained TensorFlow `Sequential` model for audio classification.
- `x_labels`: A list of class labels (e.g., `['cat', 'dog']`).

### Outputs:
- A bar plot displaying the predicted probabilities for each class.
- The original audio waveform is played for inspection.
"""

x = 'dataset/cats_dogs/test/test/dog_barking_44.wav'
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
x = tf.squeeze(x, axis=-1)
waveform = x
x = spec.get_spectrogram(x)
x = x[tf.newaxis, ...]

prediction = model(x)
x_labels = ['cat', 'dog']
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title('Prediction')
plt.show()

display.display(display.Audio(waveform, rate=16000))