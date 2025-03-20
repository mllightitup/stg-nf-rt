from ultralytics import RTDETR

# model = RTDETR('detector_weights/rtdetr-x.engine')
model = RTDETR("detector_weights/rtdetr-x.pt")
model.export(
    format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, half=True
)
