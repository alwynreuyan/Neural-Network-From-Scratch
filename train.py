from ultralytics import YOLO
model = YOLO("yolo11n.pt")

model.train(
    data = "dataset_custom.yaml",
    imgsz = 640,
    batch = 15,
    epochs = 300,
    workers = 16,
    device = "cpu"
)