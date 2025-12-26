from ultralytics import YOLO

# Load the trained model
model = YOLO("C:\\Users\\KIIT0001\\OneDrive\\Documents\\Goal\\Projects\\Empty-Shelf-Detector-main\\runs\\detect\\train\\weights\\best.pt")

# Inference on an image
results = model.predict(source="C:/Users/user/Desktop/Data Science/DL/DL Projects/SuperMarket Store Shelf Detector/test/images", save=True, imgsz=640)


