# Script to train tensorflow models. It can be used to import to Edge Impulse via BYOM (Bring Your Own Model).
# it imports the trainingdata as ususal (train/class1, train/class2, test/class1, test/class2, etc).
# Peter van Lunteren, 18 Mar 2024

# IF UPLOADING TO EDGE IMPULSE:
# Upload your trained model: 'best_model.zip'
# Model performance: only on the target device (others you can get errors with BYOM)
# Upload representative features: 'representative_features.npy'

# when uploading, it should automnatically convert to tflite and quantize it to uint8

# IF YOU MODEL IS UPLOADED TO EDGE IMPULSE:
# model input: image
# model scale: 'pixels ranging 0..255 (not normalized)' 
# resize mode: squash
# model output: classification
# output labels: classes comma separated, see contents of class_names.txt. E.g.: hippo, other

# RUN THIS SCRIPT
# conda activate "C:\Users\smart\AddaxAI_files\envs\env-tensorflow" && python "C:\Users\smart\Desktop\train-tf.py"

# set project dir to save the models in
project_dir = r"C:\Peter\projects\2024-24-TAN"

# import the necessary packages
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import zipfile
import gc
import traceback
import json
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# these parameters will be used for all models that will be trained using this script
BATCH_SIZE = 32
CROP_SIZE = 96 # normal size would be 224, but for edge deployment 96 is advised by Edge Impulse
EPOCHS = 300
PATIENCE = 30
SHUFFLE_BUFFER_SIZE = 1000 
TRAIN_SET = r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\split-on-location\without_test\full" # content must be train/class1, train/class2, test/class1, test/class2, etc

# these are the model architechtures that will be trained in a loop using the same parameters and trainset
# they must be available in the tensorflow.keras.applications library and in the load_base_model() function
MODEL_NAMES = [
    # 'efficientnetb0',             # 16 MB
    'mobilenetv3small',             # 4 MB
    # 'mobilenetv1',                # 12 MB
    # 'convnexttiny',               # 109 MB
    # 'nasnetmobile',               # 18 MB
    # 'squeezenet',                 # 23 MB
    # 'convnextsmall',              # 194 MB
    # 'efficientnetb1',             # 26 MB
    # 'mobilenetv3large',           # 12 MB
    # 'mobilenetv2',                # 9 MB
    # 'efficientnetb2'              # 31 MB
]

# init paths
TRAIN_DIR_PATH = os.path.join(TRAIN_SET, "train")
VAL_DIR_PATH = os.path.join(TRAIN_SET, "val")

# train the model architectures
def train_and_convert(MODEL_NAME):
    
    print(f"\nRUNNING MODEL_NAME: {MODEL_NAME}\n")
    
    # remove previous session memory
    train_ds = None
    val_ds = None
    tf.keras.backend.clear_session()
    gc.collect()

    # Load training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR_PATH,
        image_size=(CROP_SIZE, CROP_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Load test data
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR_PATH,
        image_size=(CROP_SIZE, CROP_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Retrieve class names from the training dataset
    class_names = train_ds.class_names
    print(f"class names (train): {class_names}")
    print(f"class names (val)  : {val_ds.class_names}")

    # Define the augmentation function
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        return image, label

    # preprocess the data
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(SHUFFLE_BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Create 'tflite' directory if it doesn't exist
    tflite_dir = os.path.join(project_dir, 'tf', MODEL_NAME)
    if not os.path.exists(tflite_dir):
        os.makedirs(tflite_dir)

    # Determine the next subdirectory name
    subdir_index = 1
    while os.path.exists(os.path.join(tflite_dir, f'train{subdir_index}')):
        subdir_index += 1

    # Create the new subdirectory
    dst_dir = os.path.join(tflite_dir, f'train{subdir_index}')
    os.makedirs(dst_dir)
    os.chdir(dst_dir)

    # Write class_names to txt file comma separated for Edge Impulse
    with open('class_names.txt', 'w') as f:
        f.write(", ".join(class_names))

    # Export settings to JSON
    settings_path = os.path.join(dst_dir, 'settings.json')
    with open(settings_path, 'w') as f:
        json.dump({
        "BATCH_SIZE": BATCH_SIZE,
        "CROP_SIZE": CROP_SIZE,
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "TRAIN_SET": TRAIN_SET
    }, f, indent=4)
    print(f"Settings saved to {settings_path}")

    # Load the chosen model
    model = load_base_model(MODEL_NAME, class_names)

    # Compile the model
    SGD = tf.keras.optimizers.SGD(learning_rate=4e-4)
    model.compile(optimizer=SGD, # SGD, # 'adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    callbacks=[checkpoint, early_stopping]
    
    # Train the model
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS,
                        callbacks=callbacks)

    # show training metrics
    plot_training_metrics(history, dst_dir)
    
    # Load the best model saved during training
    best_model = tf.keras.models.load_model('best_model.keras')

    # Evaluate the model
    evaluate_model(best_model, val_ds, class_names, dst_dir)
    
    # Save as TensorFlow SavedModel
    best_model.export('best_model')

    # Create a zip file from the saved model folder
    zip_model('best_model', 'best_model.zip')
    
    # Save a subset of the validation dataset as a .npy file for Edge Impulse
    # maximum file size is 500 MB, so let's stop collecting at 450 MB
    MAX_SIZE_MB = 400 
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
    val_data = []
    num_images = 0
    for data, _ in val_ds:  
        val_data.append(data.numpy())
        num_images += data.shape[0] 
        estimated_size = num_images * data.shape[1] * data.shape[2] * data.shape[3] * 4 # estimate based on float32
        if estimated_size > MAX_SIZE_BYTES:
            break
    val_data = np.concatenate(val_data, axis=0).astype(np.float32)
    np.save('representative_features.npy', val_data)
    print(f"\n\nSaved {num_images} images as representative features to 'representative_features.npy'")
    

# return the input tensor in the format expected by the model
def load_base_model(MODEL_NAME, class_names):
    
    # load chosen base model
    if MODEL_NAME == 'efficientnetb0':
        from tensorflow.keras.applications import EfficientNetB0
        base_model = EfficientNetB0(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'efficientnetb1':
        from tensorflow.keras.applications import EfficientNetB1
        base_model = EfficientNetB1(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'efficientnetb2':
        from tensorflow.keras.applications import EfficientNetB2
        base_model = EfficientNetB2(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'mobilenetv3small':
        from tensorflow.keras.applications import MobileNetV3Small
        base_model = MobileNetV3Small(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'mobilenetv3large':
        from tensorflow.keras.applications import MobileNetV3Large
        base_model = MobileNetV3Large(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'mobilenetv2':
        from tensorflow.keras.applications import MobileNetV2
        base_model = MobileNetV2(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'convnexttiny':
        from tensorflow.keras.applications import ConvNeXtTiny
        base_model = ConvNeXtTiny(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'nasnetmobile':
        from tensorflow.keras.applications import NASNetMobile
        base_model = NASNetMobile(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'mobilenetv1':
        from tensorflow.keras.applications import MobileNet
        base_model = MobileNet(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    elif MODEL_NAME == 'squeezenet':
        def SqueezeNet(input_shape=(224, 224, 3), include_top=True, weights='imagenet'):
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            x = fire_module(x, 64, 128)
            x = fire_module(x, 128, 256)
            x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            x = fire_module(x, 256, 256)
            x = fire_module(x, 256, 512)
            x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            x = fire_module(x, 512, 512)
            if include_top:
                x = tf.keras.layers.Conv2D(1000, (1, 1), activation='relu')(x)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x) 
            else:
                outputs = x
            model = tf.keras.Model(inputs, outputs)
            return model
        def fire_module(x, squeeze, expand):
            squeeze_layer = tf.keras.layers.Conv2D(squeeze, (1, 1), activation='relu')(x)
            expand_layer1 = tf.keras.layers.Conv2D(expand, (1, 1), activation='relu')(squeeze_layer)
            expand_layer2 = tf.keras.layers.Conv2D(expand, (3, 3), padding='same', activation='relu')(squeeze_layer)
            return tf.keras.layers.Concatenate()([expand_layer1, expand_layer2])
        base_model = SqueezeNet(input_shape=(CROP_SIZE, CROP_SIZE, 3), include_top=False)
    elif MODEL_NAME == 'convnextsmall':
        from tensorflow.keras.applications import ConvNeXtSmall
        base_model = ConvNeXtSmall(input_shape=(CROP_SIZE, CROP_SIZE, 3), weights='imagenet', include_top=False)
    
    # Freeze the base model
    base_model.trainable = False

    # Create a new model on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(class_names), activation='softmax'), 
    ])
    
    # return the model
    return model


# zip TensorFlow SavedModel directory
def zip_model(model_dir, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, model_dir))


def evaluate_model(model, ds, class_names, dst_dir):
    
    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(ds)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Predictions
    y_pred = model.predict(ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # True labels
    y_true = np.concatenate([y for x, y in ds], axis=0)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Export classification report to a text file
    report_path = os.path.join(dst_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_true, y_pred_classes, target_names=class_names))
    print(f"Classification report saved to {report_path}")

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(dst_dir, 'confusion_matrix.png'))
    
    # Normalize the confusion matrix by row (i.e., by the number of true instances)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    threshold = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > threshold else "black")
    plt.savefig(os.path.join(dst_dir, 'normalized_confusion_matrix.png'))

def plot_training_metrics(history, dst_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(dst_dir, 'training_metrics.png'))


# # Train the model
# history = model.fit(train_ds,
#                     validation_data=val_ds,
#                     epochs=EPOCHS,
#                     callbacks=callbacks)

# # show training metrics
# plot_training_metrics(history, dst_dir)

# Error: PyCapsule_New called with null pointer
# Traceback:
# Traceback (most recent call last):
#   File "C:\Users\smart\Desktop\train-tf.py", line 362, in <module>
#     train_and_convert(MODEL_NAME)
#   File "C:\Users\smart\Desktop\train-tf.py", line 177, in train_and_convert
#     plot_training_metrics(history, dst_dir)
#   File "C:\Users\smart\Desktop\train-tf.py", line 346, in plot_training_metrics
#     plt.figure(figsize=(8, 4))
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\site-packages\matplotlib\pyplot.py", line 1022, in figure
#     manager = new_figure_manager(
#               ^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\site-packages\matplotlib\pyplot.py", line 545, in new_figure_manager
#     return _get_backend_mod().new_figure_manager(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\site-packages\matplotlib\backend_bases.py", line 3521, in new_figure_manager
#     return cls.new_figure_manager_given_figure(num, fig)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\site-packages\matplotlib\backend_bases.py", line 3526, in new_figure_manager_given_figure
#     return cls.FigureCanvas.new_manager(figure, num)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\site-packages\matplotlib\backend_bases.py", line 1811, in new_manager
#     return cls.manager_class.create_with_canvas(cls, figure, num)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\site-packages\matplotlib\backends\_backend_tk.py", line 479, in create_with_canvas
#     with _restore_foreground_window_at_end():
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\contextlib.py", line 137, in __enter__
#     return next(self.gen)
#            ^^^^^^^^^^^^^^
#   File "C:\Users\smart\AddaxAI_files\envs\env-tensorflow\Lib\site-packages\matplotlib\backends\_backend_tk.py", line 43, in _restore_foreground_window_at_end
#     foreground = _c_internal_utils.Win32_GetForegroundWindow()
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ValueError: PyCapsule_New called with null pointer




# run loop
for MODEL_NAME in MODEL_NAMES:
    try:
        train_and_convert(MODEL_NAME)
    except Exception as e:
        print(f"\n\nFailed to train and convert model {MODEL_NAME}...\n\n")
        print("Error:", e)
        print("Traceback:")
        traceback.print_exc()
        print(f"\n\nContinuing with the next model...\n\n")
        continue
    