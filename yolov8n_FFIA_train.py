from ultralytics import YOLO

img_path = '/home/infish/Code/data/FFIA/images/'
pool1_path = '/home/infish/Code/data/FFIA/images/1Pool/'
pool3_path = '/home/infish/Code/data/FFIA/images/3Pool/'

# Load model
model = YOLO('yolov8n-cls.yaml')
# model = YOLO('yolov8n-cls.pt')

# Train model
result = model.train(data=img_path, imgsz=640, epochs=10, device='cuda')