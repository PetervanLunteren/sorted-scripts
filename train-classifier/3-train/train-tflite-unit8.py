




# UPDATE MARCH 2025: dit script heeft nooit goed gewerkt. Je moet niet zelf converten naar uint8, maar de EDGE IMPULSE dat laten doen. Gebruik hiervoor het script train-tf.py.













# DEBUG - code to check the model format
# conda activate ecoassistcondaenv-tensorflow && python -c "import tensorflow as tf; interpreter = tf.lite.Interpreter(model_path='model_int8.tflite'); interpreter.allocate_tensors(); input_details = interpreter.get_input_details(); output_details = interpreter.get_output_details(); print(f\"Input type: {input_details[0]['dtype']}, Output type: {output_details[0]['dtype']}\")"







# Script to train tflite uint model. This was created to make Edge Impulse compatible models for microprocessors. 
# it imports the trainingdata as ususal (train/class1, train/class2, test/class1, test/class2, etc).
# Peter van Lunteren, 30 Oct 2024

# conda activate ecoassistcondaenv-tensorflow && python "C:\Users\smart\Desktop\train-tflite-unit8.py"

# set project dir to save the models in
project_dir = r"C:\Peter\projects\2024-24-TAN"

# import the necessary packages
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import pathlib
import random
import gc
from tqdm import tqdm
import traceback
import json
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# these parameters will be used for all models that will be trained using this script
BATCH_SIZE = 32
CROP_SIZE = 224
EPOCHS = 2
PATIENCE = 5
SHUFFLE_BUFFER_SIZE = 100 # DEBUG this should be 1000
TRAIN_SET = r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\current-train-set-BOTH_TRAIN_AND_VAL_UPSAMPLED" # content must be train/class1, train/class2, test/class1, test/class2, etc
TRAIN_SET = r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\current-train-set-SMALL" # DEBUG


# these are the model architechtures that will be trained in a loop using the same parameters and trainset
# they must be available in the tensorflow.keras.applications library and in the "Load the chosen model" code block
MODEL_NAMES = [
    # 'efficientnetb0',             # 16 MB
    'mobilenetv3small',           # 4 MB
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
VAL_DIR_PATH = os.path.join(TRAIN_SET, "test")












# # DEBUG check if the converted model works

# import seaborn as sns # pip install seaborn
# from PIL import Image


# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path=r"C:\Peter\projects\2024-24-TAN\tflite\mobilenetv3small\train24\model_with_int8_quantization.tflite")
# interpreter.allocate_tensors()

# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Ensure the input type is uint8 or int8
# input_dtype = input_details[0]['dtype']
# print(f"Input data type: {input_dtype}")

# # Prepare input data
# # Loop through the subfolders and print their names
# true_labels = []
# predicted_labels = []
# test_ds_fpath = os.path.join(TRAIN_SET, "test")
# class_names = ['hippo', 'other']
# for cls_name in os.listdir(test_ds_fpath):
#     # print(f"Subfolder: {cls_name}")
#     true = cls_name
#     cls_dir = os.path.join(test_ds_fpath, cls_name)
#     # for file_name in os.listdir(cls_dir):
#     for file_name in tqdm(os.listdir(cls_dir), desc=f"Processing {cls_name}"):
#         # for file_name in files:
        
#         # print(f"file_name: {file_name}")
#         file_fpath = os.path.join(cls_dir, file_name)
#         # print(f"file_fpath: {file_fpath}")
#         if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):


#             # image_path = r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\current-train-set-BOTH_TRAIN_AND_VAL_UPSAMPLED\test\hippo\hippo-test-000001 - Copy (2).jpg"
#             image = Image.open(file_fpath)
#             input_shape = input_details[0]['shape']
#             image = image.resize((input_shape[1], input_shape[2]))
#             input_data = np.array(image, dtype=input_dtype)
#             input_data = np.expand_dims(input_data, axis=0)
#             interpreter.set_tensor(input_details[0]['index'], input_data)

#             # Run inference
#             interpreter.invoke()

#             # get prediction
#             output_data = interpreter.get_tensor(output_details[0]['index'])
#             pred_idx = np.argmax(output_data, axis=-1)
#             pred_conf = output_data[0][pred_idx]
            
#             pred_labl = class_names[pred_idx[0]]

#             # print(f"True: {true}, {pred_labl} ({pred_conf})")
#             true_labels.append(true)
#             predicted_labels.append(pred_labl)


# # Generate confusion matrix
# conf_matrix = confusion_matrix(true_labels, predicted_labels)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# # Print overall accuracy
# accuracy = accuracy_score(true_labels, predicted_labels)
# print(f"Overall Accuracy: {accuracy:.2f}")

# # Print classification report
# print("\nClassification Report:")
# print(classification_report(true_labels, predicted_labels, target_names=class_names))

# exit()











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

    # # check distribution (should be more or less balanced)
    # print(f"Training set class distribution: {get_class_distribution(train_ds)}")
    # print(f"Validation set class distribution: {get_class_distribution(val_ds)}")

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
    # train_ds = train_ds.cache().shuffle(SHUFFLE_BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE) # DEBUG removed .cache()
    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).map(augment, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE) # DEBUG removed .cache()
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) # DEBUG removed .cache()
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE) # DEBUG removed .cache()
    # normalization_layer = layers.Rescaling(1./255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # # inspect datasets
    # inspect_ds(train_ds, class_names, title = 'train', dst_dir = dst_dir)
    # inspect_ds(val_ds, class_names, title='val', dst_dir = dst_dir)

    # Create 'tflite' directory if it doesn't exist
    tflite_dir = os.path.join(project_dir, 'tflite', MODEL_NAME)
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

    # save example imgs to file # THIS SEEMS TO OVERLOAD THE GPU MEMORY
    visualise_data(TRAIN_DIR_PATH, 'train', dst_dir)
    visualise_data(VAL_DIR_PATH, 'val', dst_dir)

    # Load the chosen model
    model = load_base_model(MODEL_NAME, class_names)

    # Compile the model
    SGD = tf.keras.optimizers.SGD(learning_rate=4e-4)
    model.compile(optimizer=SGD, # SGD, # 'adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        'best_model_weights.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    callbacks=[checkpoint, early_stopping]

    # Check the unique labels in the validation dataset
    unique_labels = set()
    for _, labels in val_ds:
        unique_labels.update(labels.numpy())
    print("Unique labels in val_ds:", unique_labels)
    
    # Train the model
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS,
                        callbacks=callbacks)

    # show training metrics
    plot_training_metrics(history, dst_dir)

    # Load the best weights saved during training
    full_model = load_base_model(MODEL_NAME, class_names)
    full_model.load_weights('best_model_weights.h5')
    full_model.save('final_model.h5')

    # Optionally, load the saved model for testing
    from tensorflow.keras.models import load_model
    loaded_model = load_model('final_model.h5')

    # Evaluate the model
    evaluate_model(model, val_ds, class_names, dst_dir)
    exit()

    # Convert the model to TFLite format with uint8 quantization
    # converter1 = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Convert the model without quantization
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model2 = converter2.convert()
    with open("model_without_quantization.tflite", "wb") as f:
        f.write(tflite_model2)
    
    # SOURCE: https://medium.com/sclable/model-quantization-using-tensorflow-lite-2fe6a171a90d
    # convert a tf.Keras model to tflite model with INT8 quantization
    # Note: INT8 quantization is by default!
    converter3 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter3.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model3 = converter3.convert()
    # write the model to a tflite file as binary file
    with open("model_with_int8_quantization.tflite", "wb") as f:
        f.write(tflite_model3)
    # Note: you should see roughly 4x times reduction in the model size
    
    
    # SOURCE: https://medium.com/sclable/model-quantization-using-tensorflow-lite-2fe6a171a90d
    # create an image generator with a batch size of 1
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(TRAIN_DIR_PATH, 
                                                      target_size=(CROP_SIZE, CROP_SIZE),
                                                      batch_size=1,
                                                      shuffle=False,
                                                      class_mode='categorical')
    def represent_data_gen():
        for ind in range (len(test_generator.filenames)):
            img_with_label = test_generator.next()
            yield [np.array (img_with_label[0], type=np.float32, ndmin=2)]
    # convert a tf.Keras model to tflite model
    converter4 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter4.optimizations = [tf.lite.Optimize.DEFAULT]
    # assign the custom image generator fn to representative dataset
    converter4.representative_dataset = represent_data_gen
    tflite_model4 = converter4.convert()
    # write the model to a tflite file as binary file
    with open ("model_with_both_quantization.tflite", "wb") as f:
        f.write(tflite_model4)
        
    exit()
    
    
    # # convert to unit8 # DEBUG deze werkt!
    # print("\n\nORIGINAL CODE")
    # convert_to_uint8_org(converter, train_ds, val_ds, dst_dir, class_names)
    
    # print("\n\nWITH SUPPORTED OPS")
    # convert_to_uint8_exp(converter, train_ds, val_ds, dst_dir, class_names)
    
    print("\n\nINTEGER ONLY")
    convert_integer_only(converter, train_ds, val_ds, dst_dir, class_names)
    
    print("\n\nINTEGER WITH FLOAT FALLBACK")
    convert_integer_with_float_fallback(converter, train_ds, val_ds, dst_dir, class_names)

# experiment from https://github.com/sithu31296/PyTorch-ONNX-TFLite
def convert_integer_with_float_fallback(converter, train_ds, val_ds, dst_dir, class_names): 
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        # Use the validation dataset because there is no augmentation applied
        # unbatch and shuffle the dataset so that we can get a representative sample
        unbatched_ds = val_ds.unbatch().shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        for images, _ in unbatched_ds.take(500):  
            # Remove labels, normalize, and ensure data is in uint8 format (input range [0, 255])
            for image in images:
                yield [tf.cast(image, tf.uint8).numpy()]

    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    save_model(tflite_quant_model, 'integer_with_float_fallback', dst_dir)

# Integer only experiment from https://github.com/sithu31296/PyTorch-ONNX-TFLite
def convert_integer_only(converter, train_ds, val_ds, dst_dir, class_names):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        # Use the validation dataset because there is no augmentation applied
        # unbatch and shuffle the dataset so that we can get a representative sample
        unbatched_ds = val_ds.unbatch().shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        for images, _ in unbatched_ds.take(500):  
            # Remove labels, normalize, and ensure data is in uint8 format (input range [0, 255])
            for image in images:
                # Add batch dimension (1, height, width, channels)
                image = tf.expand_dims(image, axis=0)  # Adding batch dimension
                yield [tf.cast(image, tf.uint8).numpy()]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    save_model(tflite_quant_model, 'integer_only', dst_dir)


def convert_to_uint8_exp(converter, train_ds, val_ds, dst_dir, class_names):
    
    # Set the optimization flag
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # # Specify integer-only operations
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # # converter.target_spec.supported_types = [tf.uint8]
    
    # # generate representative dataset   
    # def representative_data_gen():
    #     for i, (image, _) in enumerate(train_ds):
    #         input_data = tf.cast(image, tf.float32)
    #         yield [input_data]
    
    
    def representative_data_gen():
        # train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)  # Adjust buffer size if necessary
        for i, (image, _) in enumerate(train_ds.take(100)):  # Limit to 100 images
            input_data = tf.cast(image, tf.float32)
            yield [input_data]
            
          
    # inspect the first image  
    for sample in representative_data_gen():
        print("Pixel range sample (should be [0, 255]):", tf.reduce_min(sample[0]).numpy(), "to", tf.reduce_max(sample[0]).numpy())
        print("sample[0].dtype", sample[0].dtype)
        print("sample[0].shape", sample[0].shape)
        break

    # inspect pixel values
    for sample in representative_data_gen():
        pass

    # Set the representative dataset
    # converter.representative_dataset = representative_data_gen

    # Convert the model with quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.inference_input_type = tf.uint8  # Set input type to uint8
    converter.inference_output_type = tf.uint8  # Set output type to uint8

    # Convert the model to TFLite
    tflite_model_int8 = converter.convert()

    # Save the int8 quantized model
    with open("model_int8.tflite", "wb") as f:
        f.write(tflite_model_int8)
    
    # # float 32 model
    # tflite_model_quant = converter.convert()
    # save_model(tflite_model_quant, 'uint8-exp-f32', dst_dir)
    
    # # uint8 model
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # tflite_model_quant = converter.convert()
    # save_model(tflite_model_quant, 'uint8-exp-ui8', dst_dir)


def save_model(model, fname, dst_dir):
    model_fpath = os.path.join(dst_dir, f'{fname}.tflite')
    with open(model_fpath, 'wb') as f:
        f.write(model)
    print(f"TFLite model with uint8 quantization is saved to '{model_fpath}'.\n\n\n")











def convert_to_uint8_org(converter, train_ds, val_ds, dst_dir, class_names):
    
    # Set the optimization flag
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # generate representative dataset   
    def representative_data_gen():
        for i, (image, _) in enumerate(train_ds):
            # for image in images:
            image_uint8 = tf.cast(image.numpy(), np.uint8)
            if i % 10000 == 0:
                print(f"Yielding image with shape: {image_uint8.shape} and dtype: {image_uint8.dtype}")
                print("Pixel range image_uint8 (should be [0, 255]):", image_uint8.numpy().min(), "to", image_uint8.numpy().max())
            yield [image_uint8]
          
    # inspect the first image  
    for sample in representative_data_gen():
        print("Pixel range sample (should be [0, 255]):", tf.reduce_min(sample[0]).numpy(), "to", tf.reduce_max(sample[0]).numpy())
        print("sample[0].dtype", sample[0].dtype)
        print("sample[0].shape", sample[0].shape)
        break

    # inspect pixel values
    for sample in representative_data_gen():
        pass

    # Set the representative dataset
    converter.representative_dataset = representative_data_gen

    # Ensure that the input and output tensors are in uint8 format
    converter.target_spec.supported_types = [tf.uint8] 

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    uint8_model_fpath = os.path.join(dst_dir, 'uint8-org.tflite')
    with open(uint8_model_fpath, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model with uint8 quantization is saved to '{uint8_model_fpath}'.\n\n\n")

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










# # DEBUG
# import tensorflow as tf

# # Load the model
# model = load_base_model('mobilenetv3small', ['hippo', 'other'])
# # model = tf.saved_model.load(r"C:\Peter\projects\2024-24-TAN\tflite\mobilenetv3small\train7\uint8-org.tflite")

# # Print model summary to check the input/output layers
# print(model.summary())

# # Get the input and output tensors
# input_tensor = model.input
# output_tensor = model.output

# # Print the input tensor details
# print("\nModel input tensor:")
# print(input_tensor)

# # Print the output tensor details
# print("\nModel output tensor:")
# print(output_tensor)

# # Inspect the layers
# print("\nModel layers:")
# for layer in model.layers:
#     print(layer.name, layer.input_shape, layer.output_shape)


# exit()
# DEBUG

# Traceback (most recent call last):
#   File "C:\Users\smart\Desktop\train-tflite-unit8.py", line 706, in <module>
#     input_signature = model.signatures['serving_default']
# AttributeError: 'Sequential' object has no attribute 'signatures'


# visualise the images
def visualise_data(dir_path, title, dst_dir, num_images=18):
    
    # Get the list of class subdirectories
    class_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # Initialize the plot
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.flatten()

    # Counter for the subplot index
    index = 0
    half_num_images = num_images // 2

    # Iterate through each class subdirectory
    for class_dir in class_dirs:
        class_path = os.path.join(dir_path, class_dir)
        
        # Get the list of image files in the class subdirectory
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(image_files)
        
        # Iterate through each image file
        for image_file in image_files:
            if index >= half_num_images:
                break
            image_path = os.path.join(class_path, image_file)
            img = mpimg.imread(image_path)
            axes[index].imshow(img)
            axes[index].set_title(class_dir)
            axes[index].axis('off')
            index += 1

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(dst_dir, f'imgs-{title}.png'))
    # plt.show()
    

# def inspect_ds(dataset, class_names, title, dst_dir, num_images=18):
    
#     print("\n\n")
#     print(f"Inspecting   : {title}")
    
#     # # inspect values
#     # for images, labels in dataset.take(20):
#     #     print("Pixel range  :", images[0].numpy().min(), "to", images[0].numpy().max())

#     # # inspect values
#     # for images, labels in dataset.take(1):
#     #     print("Images shape :", images.shape)
#     #     print("Labels shape :", labels.shape)
#     #     print("Labels       :", labels.numpy()[0:10])
#     # print("\n")
    
#     # # Set up the figure and axes
#     # plt.figure(figsize=(12, 7))
#     # plt.suptitle(title, fontsize=16)
#     # images_shown = 0

#     # # Loop over the dataset until the required number of images are shown
#     # for images, labels in dataset:
#     #     for i in range(len(images)):
#     #         if images_shown >= num_images:
#     #             break
#     #         ax = plt.subplot(3, 6, images_shown + 1)  # Adjust the grid as needed
#     #         plt.imshow(images[i].numpy().astype("uint8"))
#     #         plt.title(class_names[labels[i].numpy()])
#     #         plt.axis("off")
#     #         images_shown += 1
#     #     if images_shown >= num_images:
#     #         break
#     # plt.savefig(os.path.join(dst_dir, f'imgs-{title}.png'))

#     # Gather all images and labels in a list
#     all_images = []
#     all_labels = []
#     for images, labels in dataset:
#         for i in range(len(images)):
#             all_images.append(images[i])
#             all_labels.append(labels[i])

#     # Shuffle the images and labels in unison
#     combined = list(zip(all_images, all_labels))
#     random.shuffle(combined)
#     all_images, all_labels = zip(*combined)

#     # Set up the figure and axes
#     plt.figure(figsize=(12, 7))
#     plt.suptitle(title, fontsize=16)
#     images_shown = 0

#     # Display the first num_images images
#     for images_shown in range(num_images):
#         ax = plt.subplot(3, 6, images_shown + 1)
#         plt.imshow(all_images[images_shown].numpy())
#         plt.imshow(all_images[images_shown].numpy().astype("uint8"))
#         # plt.imshow((all_images[images_shown].numpy() * 255).astype("uint8"))
#         plt.title(class_names[all_labels[images_shown].numpy()])
#         plt.axis("off")

#     # Save the figure
#     plt.savefig(os.path.join(dst_dir, f'imgs-{title}.png'))


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

# # class to quantize the model
# class QuantModel:
#     def __init__(self, model=tf.keras.Model, data=[]):
#         self.data = data
#         self.model = model

#     def quant_model_int8(self):
#         converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
#         converter.representative_dataset = self.representative_data_gen
#         converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#         converter.inference_input_type = tf.int8  # or tf.uint8
#         converter.inference_output_type = tf.int8  # or tf.uint8
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         tflite_model_quant = converter.convert()
#         open("converted_model2.tflite", 'wb').write(tflite_model_quant)
#         return tflite_model_quant

#     def convert_tflite_no_quant(self):
#         converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
#         tflite_model = converter.convert()
#         open("converted_model.tflite", 'wb').write(tflite_model)
#         return tflite_model

#     def representative_data_gen(self):
#         for input_value, _ in self.data:
#             yield [input_value]

# # class to quantize the model with slight adjustment to test the effect
# class QuantModel2:
#     def __init__(self, model=tf.keras.Model, data=[]):
#         self.data = data
#         self.model = model

#     def quant_model_int8(self):
#         converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
#         converter.representative_dataset = self.representative_data_gen
#         converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#         # converter.inference_input_type = tf.int8  # or tf.uint8
#         # converter.inference_output_type = tf.int8  # or tf.uint8
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         tflite_model_quant = converter.convert()
#         open("converted_model2.tflite", 'wb').write(tflite_model_quant)
#         return tflite_model_quant

#     def convert_tflite_no_quant(self):
#         converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
#         tflite_model = converter.convert()
#         open("converted_model.tflite", 'wb').write(tflite_model)
#         return tflite_model

#     def representative_data_gen(self):
#         for input_value, _ in self.data:
#             yield [input_value]

# def get_dataset_size(dataset):
#     size = 0
#     for _ in dataset:
#         size += 1
#     return size

# # get the first, middle and last images from a dataset
# def get_first_middle_last_images(ds, sample_size=500):
#     unbatched_ds = ds.unbatch()
#     dataset_size = len(list(unbatched_ds))
#     first_sample = unbatched_ds.take(sample_size)
#     middle_sample = unbatched_ds.skip((dataset_size // 2) - (sample_size // 2)).take(sample_size)
#     last_sample = unbatched_ds.skip(dataset_size - sample_size).take(sample_size)
#     combined_ds = first_sample.concatenate(middle_sample).concatenate(last_sample).batch(BATCH_SIZE)
#     return combined_ds

# # DEBUG deze pakt de quantmodel met converter.inference_input_type = tf.int8
# # Deze lijkt te werken: kies input 0..255 of -218..127. Lijkt me dan dat je -128..127 moet kiezen.
# # exporteren gaat ook goed.
# # fully_quantize: 0, inference_type: 6, input_inference_type: INT8, output_inference_type: INT8
# # als ik -128..127 als input pak, dan geeft ie 39.2%% hippo-hippo-accuracy, als ik 0..255 pak, dan geeft ie 36.1% hippo-hippo-accuracy
# def h5_to_int8_option2(h5_model_fpath, base_model_name, class_names):
#     model = load_base_model(base_model_name, class_names)
#     model.load_weights(h5_model_fpath)

#     # Load training data
#     train_ds_int8_1 = tf.keras.preprocessing.image_dataset_from_directory(
#         TRAIN_DIR_PATH,
#         image_size=(CROP_SIZE, CROP_SIZE),
#         batch_size=BATCH_SIZE,
#         shuffle=True
#     )
    
#     # Instantiate QuantModel
#     quant_model = QuantModel(model=model, data=get_first_middle_last_images(train_ds_int8_1))
    
#     # Perform quantization
#     tflite_quant_model = quant_model.quant_model_int8()
    
#     # Save the TFLite model
#     tflite_model_quant_file = os.path.join(os.path.dirname(h5_model_fpath), 'int8_option2.tflite')
#     with open(tflite_model_quant_file, 'wb') as f:
#         f.write(tflite_quant_model)

#     print(f"\nTFLite model with int8 quantization is saved to '{tflite_model_quant_file}'.\n\n")
    
#     del train_ds_int8_1
#     gc.collect()




# # DEBUG deze pakt quant model zonder converter.inference_input_type = tf.int8
# # fully_quantize: 0, inference_type: 6, input_inference_type: FLOAT32, output_inference_type: FLOAT32
# # Deze geeft veel meer latency als optie 2...

# # als ik -128..127 als input pak, dan geeft ie 31.6% hippo-hippo-accuracy
# # als ik 0..255 pak, dan geeft ie 47.3% hippo-hippo-accuracy
# # als ik 0..1 (not normalized) pak, dan geeft ie 18.9% hippo-hippo-accuracy (very bad!)
# # als ik RGB -> BGR pak, dan geeft ie x% hippo-hippo-accuracy 
# def h5_to_int8_option3(h5_model_fpath, base_model_name, class_names):
#     model = load_base_model(base_model_name, class_names)
#     model.load_weights(h5_model_fpath)

#     # Load training data
#     train_ds_int8_2 = tf.keras.preprocessing.image_dataset_from_directory(
#         TRAIN_DIR_PATH,
#         image_size=(CROP_SIZE, CROP_SIZE),
#         batch_size=BATCH_SIZE,
#         shuffle=True
#     )
    
#     # Instantiate QuantModel
#     quant_model = QuantModel2(model=model, data=get_first_middle_last_images(train_ds_int8_2))
    
#     # Perform quantization
#     tflite_quant_model = quant_model.quant_model_int8()
    
#     # Save the TFLite model
#     tflite_model_quant_file = os.path.join(os.path.dirname(h5_model_fpath), 'int8_option3.tflite')
#     with open(tflite_model_quant_file, 'wb') as f:
#         f.write(tflite_quant_model)

#     print(f"\nTFLite model with int8 quantization is saved to '{tflite_model_quant_file}'.\n\n")
    
#     del train_ds_int8_2
#     gc.collect()


# # # h5_to_int8_option1(r"C:\Peter\projects\2024-24-TAN\tflite\efficientnetb0\train3\best_model_weights.h5", 'efficientnetb0', ['hippo', 'other'])
# # h5_to_int8_option2(r"C:\Peter\projects\2024-24-TAN\tflite\efficientnetb0\train1\best_model_weights.h5", 'efficientnetb0', ['hippo', 'other'])
# # h5_to_int8_option3(r"C:\Peter\projects\2024-24-TAN\tflite\efficientnetb0\train1\best_model_weights.h5", 'efficientnetb0', ['hippo', 'other'])

# # exit()








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
    