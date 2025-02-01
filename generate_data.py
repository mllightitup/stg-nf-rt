from ultralytics import YOLO
import json

model = YOLO("yolo11n-pose.pt")

results = model.track("crowd.mp4", imgsz=1920, persist=True, classes=[0])

tracked_person = {}

for i, frame in enumerate(results):
    for track_id, keypoint, box, score in zip(
        frame.boxes.id.int().cpu().tolist(),
        frame.keypoints.data.cpu().reshape(-1, 51).tolist(),
        frame.boxes.xywh.cpu().tolist(),
        frame.boxes.conf.cpu().tolist()
    ):
        tracked_person.setdefault(track_id, {})[i] = {
            'keypoints': keypoint,
            'scores': score,
            'boxes': box
        }


with open('aboba.json', 'w') as json_file:
    json.dump(tracked_person, json_file, indent=4)