
# script to convert YOLOv8 cls model to TFlite uint8 model campatible with Edge Impusle
# 23 Oct, 2024 - Peter van Lunteren

# this script assumes yolov8 training has just finished. The model is in it wieghts folder and the test data is still the test folder

####### STEP 1: cd to model dir - for example: cd C:\Peter\projects\2024-24-TAN\runs\classify\train2

####### STEP 2: run script
# conda activate ecoassistcondaenv-tensorflow && python "C:\Users\smart\Desktop\convert-yolov8-to-tflite-unit8.py"

import tensorflow as tf 
import numpy as np
import os
from PIL import Image
import subprocess

# step 1: convert yolov8 cls model to tflite
command = ["C:\\Users\\smart\\miniforge3\\condabin\\conda.bat",  "activate", "ecoassistcondaenv-pytorch", "&&", "C:\\Users\\smart\\miniforge3\\envs\\ecoassistcondaenv-pytorch\\Scripts\\yolo.exe", "export", "model=weights\\best.pt", "format=tflite"]
subprocess.run(command, check=True)

# init model paths
train_data_dir = r"C:\Peter\training-utils\current-train-set-NON_DUPLICATES\train"
saved_model_dir = os.path.join("weights", "best_saved_model")
dst_model_path = os.path.join("weights", "best_uint8.tflite")
BATCH_SIZE = 32 # for the representative dataset

# Convert to uint8 using a representative dataset
image_height, image_width = 224, 224 # this is the default for all yolov8-cls models (n, s, m, l, x)

# define representative dataset
def representative_dataset():
    # Load training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        image_size=(image_height, image_width),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # yield representative dataset
    for i, (image, _) in enumerate(train_ds):
        # for image in images:
        image_uint8 = tf.cast(image.numpy(), np.uint8)
        if i % 10000 == 0:
            print(f"Yielding image with shape: {image_uint8.shape} and dtype: {image_uint8.dtype}")
            print("Pixel range image_uint8 (should be [0, 255]):", image_uint8.numpy().min(), "to", image_uint8.numpy().max())
        yield [image_uint8]

# open the model and save it with a serving default signature
saved_model_dir = r"weights\best_saved_model"
model = tf.keras.models.load_model(saved_model_dir)
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def serve_model(input_tensor):
    return model(input_tensor)
tf.saved_model.save(model, saved_model_dir, signatures={'serving_default': serve_model})
print(list(model.signatures.keys()))

# print dataset
for sample in representative_dataset():
    pass

# Step 3: Convert the model to uint8 using the representative dataset
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.uint8]

# Perform the conversion
quantized_model = converter.convert()

# Step 4: Save the quantized model
with open(dst_model_path, "wb") as f:
    f.write(quantized_model)

print(f"\n\n\n\nQuantized model saved to: {dst_model_path}")
