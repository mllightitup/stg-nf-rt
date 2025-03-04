import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import RTDETR, YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation
import supervision as sv


import random

import cv2
import numpy as np
import torch

from models.STG_NF.model_pose import STG_NF
from models.training import InferenceSTG
from custom_deque import Tracker
from args import init_parser, init_sub_args

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

# Параметр, определяющий длину истории для трекера:
max_history = 24

parser = init_parser()
args = parser.parse_args()

if args.seed == 999:  # Record and init seed
    args.seed = torch.initial_seed()
    np.random.seed(0)
else:
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(0)

args, model_args = init_sub_args(args)

args.dataset = "ShanghaiTech"
args.checkpoint = r"C:\Users\Grishin\PycharmProjects\stg-nf-rt\STG-NF\data\exp_dir\ShanghaiTech\Mar04_1336\Mar04_1337__checkpoint.pth.tar"
# args.checkpoint = r"checkpoints/ShanghaiTech_85_9.tar"

pretrained = vars(args).get("checkpoint", None)

model_args = {
    "pose_shape": (2, 24, 18),
    "hidden_channels": args.model_hidden_dim,
    "K": args.K,
    "L": args.L,
    "R": args.R,
    "actnorm_scale": 1.0,
    "flow_permutation": args.flow_permutation,
    "flow_coupling": "affine",
    "LU_decomposed": True,
    "learn_top": False,
    "edge_importance": args.edge_importance,
    "temporal_kernel_size": args.temporal_kernel,
    "strategy": args.adj_strategy,
    "max_hops": args.max_hops,
    "device": args.device,
}

model = STG_NF(**model_args)
inference = InferenceSTG(args, model)

inference.load_checkpoint(pretrained)


ORDER = torch.tensor(
    [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
    dtype=torch.long,
    device=device,
)
NORM_FACTOR = torch.tensor((856, 480, 1), dtype=torch.float32, device=device)


def keypoints17_to_coco18(kps17):
    """
    Допустим, kps17: (B, max_history, 17, 3)
    Добавляем шейный keypoint, переставляем в coco18.
    """
    # neck = среднее keypoints 5 и 6 в 17-точечной схеме:
    neck = kps17[..., [5, 6], :].mean(dim=-2, keepdim=True)
    kps18 = torch.cat([kps17, neck], dim=-2)  # (B, max_history, 18, 3)
    return kps18.index_select(dim=-2, index=ORDER)  # reorder


def normalize_pose(pose_data, symm_range=False):
    """
    pose_data: (B, max_history, 18, 3), B - кол-во треков
    """
    data = pose_data / NORM_FACTOR
    if symm_range:
        data[..., :2] = 2.0 * data[..., :2] - 1.0

    mean_xy = data[..., :2].mean(dim=(1, 2), keepdim=True)
    std_y = data[..., 1].std(dim=(1, 2), keepdim=True).unsqueeze(-1)
    data[..., :2] = (data[..., :2] - mean_xy) / std_y
    return data


# ----------------------------------------------------
# YOLO/RTDETR + ViTPose
# ----------------------------------------------------
yolo_model = YOLO(r"C:\Users\Grishin\PycharmProjects\stg-nf-rt\yolo_models\yolo12n.pt")

pose_checkpoint = "usyd-community/vitpose-plus-small"
pose_model = VitPoseForPoseEstimation.from_pretrained(
    pose_checkpoint, torch_dtype=dtype, local_files_only=True
).to(device)
pose_processor = AutoProcessor.from_pretrained(
    pose_checkpoint, use_fast=False, local_files_only=True
)

# Инициализируем наш трекер:
tracker = Tracker(max_history=max_history, device=device)


# ----------------------------------------------------
# Препроцессинг боксов и изображений
# ----------------------------------------------------
def preprocess_boxes(
    boxes_xyxy: np.ndarray, crop_height=256, crop_width=192, padding_factor=1.25
):
    aspect_ratio = crop_width / crop_height
    x_min, y_min, x_max, y_max = boxes_xyxy.T
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Приводим к нужному соотношению сторон:
    height = np.where(width > height * aspect_ratio, width / aspect_ratio, height)
    width = np.where(width < height * aspect_ratio, height * aspect_ratio, width)
    height *= padding_factor
    width *= padding_factor

    new_x_min = x_center - width / 2
    new_y_min = y_center - height / 2
    new_x_max = new_x_min + width
    new_y_max = new_y_min + height
    return np.stack([new_x_min, new_y_min, new_x_max, new_y_max], axis=1)


def preprocess_image(
    image: np.ndarray, boxes_xyxy: np.ndarray, crop_height, crop_width, mean, std
):
    boxes = preprocess_boxes(boxes_xyxy, crop_height, crop_width)
    boxes = torch.from_numpy(boxes).float().to(device)

    image_tensor = (
        torch.from_numpy(image.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    crops = torchvision.ops.roi_align(
        image_tensor, [boxes], (crop_height, crop_width), spatial_scale=1
    )
    crops = crops.to(dtype)

    tensor_mean = torch.tensor(mean, dtype=crops.dtype, device=device).view(1, 3, 1, 1)
    tensor_std = torch.tensor(std, dtype=crops.dtype, device=device).view(1, 3, 1, 1)
    crops = (crops / 255.0 - tensor_mean) / tensor_std

    return {"pixel_values": crops}, boxes


def postprocess_keypoints(
    heatmaps: torch.Tensor, boxes: torch.Tensor, crop_height, crop_width
):
    b_size, num_keypoints, _, _ = heatmaps.shape
    heatmaps = F.interpolate(
        heatmaps, size=(crop_height, crop_width), mode="bilinear", align_corners=True
    )
    heatmaps_reshaped = heatmaps.view(b_size, num_keypoints, -1)
    scores, indices = heatmaps_reshaped.max(dim=-1)

    keypoints_x = indices % crop_width
    keypoints_y = indices // crop_width

    box_x1, box_y1, box_x2, box_y2 = boxes.split(1, dim=-1)
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    keypoints_x = keypoints_x.float() * box_width / crop_width + box_x1
    keypoints_y = keypoints_y.float() * box_height / crop_height + box_y1

    keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)
    return keypoints, scores


def value_to_color(value):
    # Нормализуем значение в диапазон [0, 1]
    min_value, max_value = (
        -10,
        0,
    )  # Например, предположим, что минимальные и максимальные значения value в этом диапазоне
    normalized_value = np.clip((value - min_value) / (max_value - min_value), 0, 1)

    # Используем normalized_value для создания цвета: красный - зеленый
    red = int(normalized_value * 255)  # Чем меньше значение, тем краснее
    green = int((1 - normalized_value) * 255)  # Чем больше значение, тем зеленее
    return 0, red, green  # Цвет в формате (B, G, R)


def visualize_output(
    image: np.ndarray,
    boxes: torch.Tensor,
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    person_ids: np.ndarray,  # добавили IDs
    track2normal: dict,  # словарь track_id → normality_score
    conf_threshold=0.01,
):
    mask = scores < conf_threshold
    keypoints[mask] = 0
    kp = sv.KeyPoints(xy=keypoints.cpu().numpy(), confidence=scores.cpu().numpy())
    detections = sv.Detections(
        xyxy=boxes.cpu().numpy(),
        # class_id=np.zeros(boxes.shape[0]),
    )

    annotated = image.copy()

    edge_annotator = sv.EdgeAnnotator(color=sv.Color.YELLOW, thickness=1)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.ROBOFLOW, radius=2)
    # box_annotator = sv.BoxAnnotator(color=sv.Color.WHITE, thickness=2)

    annotated = edge_annotator.annotate(annotated, key_points=kp)
    annotated = vertex_annotator.annotate(annotated, key_points=kp)
    # annotated = box_annotator.annotate(annotated, detections=detections)

    # Вывод normality_scores над боксами
    for i, box in enumerate(detections.xyxy):
        tid = int(person_ids[i])
        if tid in track2normal:
            nscore = float(track2normal[tid])
            x1, y1, x2, y2 = box
            cv2.putText(
                annotated,
                f"{nscore:.2f}",
                (int(x1), max(int(y1) - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"{nscore:.2f}",
                (int(x1), max(int(y1) - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                value_to_color(nscore),
                1,
            )
    return annotated


def process_frame(frame: np.ndarray, frame_index: int):
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yolo_results = yolo_model.track(frame, verbose=False, classes=[0], persist=True)[0]
    boxes = yolo_results.boxes.xyxy.cpu().numpy()
    confs = yolo_results.boxes.conf.cpu().numpy()
    classes = yolo_results.boxes.cls.cpu().numpy()

    # Проверяем track IDs
    if yolo_results.boxes.id is not None:
        ids = yolo_results.boxes.id.cpu().numpy()
    else:
        return frame, {}, None

    mask = (classes == 0) & (confs > 0.01)
    if mask.sum() == 0:
        return frame, {}, None

    person_det = np.concatenate([boxes, confs[:, None], classes[:, None]], axis=1)[mask]
    person_ids = ids[mask]
    boxes_xyxy = person_det[:, :4]

    crop_height = pose_processor.size["height"]
    crop_width = pose_processor.size["width"]

    inputs, boxes_tensor = preprocess_image(
        frame,
        boxes_xyxy,
        crop_height,
        crop_width,
        mean=pose_processor.image_mean,
        std=pose_processor.image_std,
    )
    inputs["dataset_index"] = torch.zeros(
        boxes_tensor.shape[0], dtype=torch.int64, device=device
    )

    with torch.no_grad():
        pose_outputs = pose_model(**inputs)

    keypoints, scores = postprocess_keypoints(
        pose_outputs.heatmaps, boxes_tensor, crop_height, crop_width
    )
    detections_keypoints = torch.cat([keypoints, scores.unsqueeze(-1)], dim=-1)

    tids_tensor = torch.from_numpy(person_ids).long().to(device)
    b_conf_tensor = torch.from_numpy(person_det[:, 4]).float().to(device)
    tracker.update(tids_tensor, detections_keypoints, b_conf_tensor)

    # Данные для JSON
    frame_data = {}
    for i, pid in enumerate(person_ids):
        kp_list = detections_keypoints[i].cpu().numpy().flatten().tolist()
        box_conf = float(person_det[i, 4])
        frame_data[int(pid)] = {"keypoints": kp_list, "score": box_conf}

    normality_scores = None
    if frame_index >= max_history - 1:
        pose_tensor, conf_tensor, union_ids = tracker.build_tensor()
        if pose_tensor is not None:
            kps_coco18 = keypoints17_to_coco18(pose_tensor)
            kps_norm = normalize_pose(kps_coco18)
            kps_final = kps_norm.permute(0, 3, 1, 2)
            normality_scores = inference.test_real_time(kps_final, conf_tensor)

            # Формируем map track_id → normality_score
            track2normal = {}
            # print(normality_scores)
            if union_ids.shape[0] == 1:
                # Всего один трек
                track2normal[int(union_ids.item())] = float(normality_scores.item())
            else:
                # Несколько треков
                for i, uid in enumerate(union_ids):
                    track2normal[int(uid.item())] = float(normality_scores[i].item())

            annotated_frame = visualize_output(
                frame,
                boxes_tensor,
                keypoints,
                scores,
                person_ids,  # <-- пробрасываем
                track2normal,  # <-- пробрасываем
            )
            # annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            return annotated_frame, frame_data, normality_scores

    return frame, frame_data, normality_scores


def main(input_source, output_file="annotated_video.mp4", json_output="results.json"):
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # global_results = {}
    frame_id = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, frame_data, normal_scores = process_frame(frame, frame_id)
        out.write(annotated_frame)
        cv2.imshow("Video", annotated_frame)

        # Сохраняем JSON-данные
        # for pid, data in frame_data.items():
        #     if pid not in global_results:
        #         global_results[pid] = {}
        #     global_results[pid][frame_id] = data

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # with open(json_output, "w") as f:
    #     # Для удобства ремапим ID в 1..N
    #     new_data = {
    #         new_key: global_results[old_key]
    #         for new_key, old_key in enumerate(sorted(global_results), start=1)
    #     }
    #     json.dump(new_data, f)
    #
    # print("JSON results saved to:", json_output)


if __name__ == "__main__":
    main(
        r"C:\Users\Grishin\Desktop\OneDrive-2025-02-22\shanghaitech.tar\shanghaitech\testing\videos\01_0014.mp4",
        # "08_016_results.json",
    )

    # path_testing = r"C:\Users\Grishin\Desktop\OneDrive-2025-02-22\shanghaitech.tar\shanghaitech\testing\videos"
    # videos = os.listdir(path_testing)
    # for video in tqdm(videos):
    #     if yolo_model.predictor is not None:
    #         yolo_model.predictor.trackers[0].reset()
    #     main(
    #         path_testing + "\\" + video,
    #         rf"C:\Users\Grishin\Desktop\OneDrive-2025-02-22\shanghaitech.tar\shanghaitech\testing\rtdetr640_vitpose-plus-small\{video.split('.')[0]}_alphapose_tracked_person.json",
    #     )
