from ultralytics import RTDETR

# model = RTDETR('detector_weights/rtdetr-x.engine')
model = RTDETR("detector_weights/rtdetr-x.pt")
model.export(
    format="engine", imgsz=640, dynamic=True, verbose=False, batch=8, half=True
)

# results = model.track(
#     source=r"F:\shanghaitech\training\videos\08_016.avi",
#     stream=True,
#     show=True,
#     classes=[0],
#     line_width=1,
#     half=True,
#     tracker="bytetrack.yaml",
# )
#
# for result in results:
#     pass
