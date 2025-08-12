from ultralytics import YOLO
import torch

print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")

# Load a model
# model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8n-cls.pt', device = 'gpu')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data=r'C:\Users\smart\Desktop\crops-trainset', epochs=1, imgsz=64)