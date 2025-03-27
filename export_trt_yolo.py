from ultralytics import RTDETR

model = RTDETR("detector_weights/rtdetr-l.pt")
model.export(format="engine", dynamic=False, verbose=False, batch=1, half=True)
