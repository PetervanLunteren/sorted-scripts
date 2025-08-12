
# conda activate ecoassistcondaenv-pytorch && python "C:\Users\smart\Desktop\models for Thomas\yolov8n2uint8.py"




# # THIS WORKS
# # I convert the yolov8n model with the code below
# import pathlib
# import torch
# import os
# from ultralytics import YOLO
# model = YOLO(r"C:\Users\smart\Desktop\yolo-model\yolov8n_saved_model\yolov8n.pt")
# model.export(format="tflite", int8=True)

# # that results in a couple of models. The yolov8n_full_integer_quant.pt works in Egde Impulse (make sure to set input to 0..255). 
# # tflite_model = YOLO(r"C:\Users\smart\Desktop\yolo-model\yolov8n_saved_model\yolov8n_float32.tflite", task='classify')
# # tflite_model = YOLO(r"C:\Users\smart\Desktop\yolo-model\yolov8n_saved_model\yolov8n_float16.tflite", task='classify')
# # tflite_model = YOLO(r"C:\Users\smart\Desktop\yolo-model\yolov8n_saved_model\yolov8n_int8.tflite", task='classify')
# # tflite_model = YOLO(r"C:\Users\smart\Desktop\yolo-model\yolov8n_saved_model\yolov8n_full_integer_quant.tflite", task='classify')
# tflite_model = YOLO(r"C:\Users\smart\Desktop\yolo-model\yolov8n_saved_model\yolov8n_integer_quant.tflite", task='classify')
# results = tflite_model(r"C:\Users\smart\Desktop\yolo-model\bus.jpg", imgsz=224)
# for result in results:
#     if result.probs:
#         print("Probabilities:")
#         for class_id, prob in enumerate(result.probs.data):
#             class_name = result.names.get(class_id, f"Class {class_id}")
#             print(f"\t{class_name}: {prob:.4f}")


# NEW APPROACH: (https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2)
from ultralytics import YOLO
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# # export to onnx
# model = YOLO('2-yolov8-int8-false.pt')
# model.export(format = 'onnx')
# exit()



# # proprocess the onnx model
# # seems to work only if you put input to 0..255, but then the test is bad.  
# pip install onnxruntime
# python -m onnxruntime.quantization.preprocess --input yolov8n.onnx --output preprocessed.onnx


## https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu
# python run.py --input_model 1-yolov8-int8-true-preprocessed.onnx --output_model 1-yolov8-int8-true-preprocessed.quant.onnx --calibrate_dataset ./test_images/
# python run.py --input_model 2-yolov8-int8-false-preprocessed.onnx --output_model 2-yolov8-int8-false-preprocessed.quant.onnx --calibrate_dataset ./test_images/





# dynamic quantization: doesnt seem to work in EI, even not with the representative data in .npy file
from onnxruntime.quantization import quantize_dynamic, QuantType
model_fp32 = '2-yolov8-int8-false-preprocessed.onnx'
model_int8 = '2-yolov8-int8-false-preprocessed-dynamic_quantized.onnx'
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)




# create representative data
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
image_dir = "representative_data"
image_size = (224, 224)
output_file = "representative_data.npy"
# image_data = []
# image_files = [os.path.join(root, filename) for root, dirs, files in os.walk(image_dir) for filename in files if filename.lower().endswith((".jpg", ".png", ".jpeg"))]
# for filepath in tqdm(image_files, desc="Processing images"):
#     img = Image.open(filepath).convert("RGB")
#     img = img.resize((224, 224))
#     img_array = np.array(img, dtype=np.float32) / 255.0
#     img_array = np.transpose(img_array, (2, 0, 1))
#     img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 3, 224, 224)
#     image_data.append(img_array)
image_data = np.stack(image_data)
np.save(output_file, image_data)
loaded_data = np.load("representative_data.npy")
print(loaded_data.shape)  # Should match (num_images, 1, 3, 224, 224)
import numpy as np

# # Load the .npy file
# file_path = "representative_data.npy"
# data = np.load(file_path)

# # Check the shape of the entire array
# print(f"Shape of entire array: {data.shape}")

# # Check the shape of each element (if the array contains multiple elements)
# for i, element in enumerate(data):
#     print(f"Shape of element {i}: {element.shape}")

# exit()

#  static quantization
import numpy as np
import cv2
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat
class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.idx = 0
        self.input_name = "images"

    def preprocess(self, frame):
        # Same preprocessing that you do before feeding it to the model
        frame = cv2.imread(frame)
        X = cv2.resize(frame, (224, 224))
        image_data = np.array(X).astype(np.float32) / 255.0  # Normalize to [0, 1] range
        image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
        return image_data

    def get_next(self):
        # method to iterate through the data set
        if self.idx >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.idx]
        input_data = self.preprocess(image_path)
        self.idx += 1
        return {self.input_name: input_data}
calibration_image_paths = [r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\current-train-set-SMALL\test\hippo\seq000001-img000001-non-local.JPG",
                           r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\current-train-set-SMALL\test\hippo\seq000016-img000022-non-local.JPG",
                           r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\current-train-set-SMALL\test\other\seq000007-img000008-non-local.JPG",
                           r"C:\Peter\projects\2024-24-TAN\imgs\training-data-sets\current-train-set-SMALL\test\other\seq000016-img000018-non-local.JPG"] 
calibration_data_reader = ImageCalibrationDataReader(calibration_image_paths)
quantize_static(model_fp32, "2-yolov8-int8-false-preprocessed-static_quantized.onnx",
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantFormat.QDQ,
                nodes_to_exclude=['/model.22/Concat_3', '/model.22/Split', '/model.22/Sigmoid'
                                 '/model.22/dfl/Reshape', '/model.22/dfl/Transpose', '/model.22/dfl/Softmax', 
                                 '/model.22/dfl/conv/Conv', '/model.22/dfl/Reshape_1', '/model.22/Slice_1',
                                 '/model.22/Slice', '/model.22/Add_1', '/model.22/Sub', '/model.22/Div_1',
                                  '/model.22/Concat_4', '/model.22/Mul_2', '/model.22/Concat_5'],
                per_channel=False,
                reduce_range=True,)

