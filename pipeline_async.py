import asyncio
import random
import time

import torch
import torchvision
import torch.nn.functional as F

import cv2
import numpy as np
import supervision as sv

from ultralytics import RTDETR
from transformers import AutoProcessor, VitPoseForPoseEstimation
from STG_NF.args import init_sub_args, init_parser
from STG_NF.models.STG_NF.model_pose import STG_NF
from inferense_stg import InferenceSTG
from buffer import BufferManager

# ------------------------------
# Set device and dtype
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

# ------------------------------
# History length for the tracker
# ------------------------------
max_history = 24

# ------------------------------
# Parse args and set up seeds
# ------------------------------
parser = init_parser()
args = parser.parse_args()

if args.seed == 999:
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
args.checkpoint = (
    r"STG_NF/checkpoints/Mar15_2259__checkpoint.pth.tar"
)

pretrained = vars(args).get("checkpoint", None)

# ------------------------------
# Model arguments for STG_NF
# ------------------------------
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

# ------------------------------
# Build and load the normality model
# ------------------------------
normality_model = STG_NF(**model_args).eval().to(device)
stg_inference = InferenceSTG(args, normality_model)
stg_inference.load_checkpoint(pretrained)

# ------------------------------
# Helper reorder and normalization
# ------------------------------
ORDER = torch.tensor(
    [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
    dtype=torch.long,
    device=device,
)
NORM_FACTOR = torch.tensor((856, 480, 1), dtype=torch.float32, device=device)


def keypoints17_to_coco18(kps17):
    """
    kps17: (B, max_history, 17, 3)
    Add neck keypoint, reorder to coco18.
    """
    neck = kps17[..., [5, 6], :].mean(dim=-2, keepdim=True)
    kps18 = torch.cat([kps17, neck], dim=-2)
    return kps18.index_select(dim=-2, index=ORDER)


def normalize_pose(pose_data, symm_range=False):
    """
    pose_data: (B, max_history, 18, 3)
    """
    data = pose_data / NORM_FACTOR
    if symm_range:
        data[..., :2] = 2.0 * data[..., :2] - 1.0

    mean_xy = data[..., :2].mean(dim=(1, 2), keepdim=True)
    std_y = data[..., 1].std(dim=(1, 2), keepdim=True).unsqueeze(-1)
    data[..., :2] = (data[..., :2] - mean_xy) / std_y
    return data


# ------------------------------
# Prepare YOLO/RTDETR and ViTPose
# ------------------------------
yolo_model = RTDETR(r"detector_weights/rtdetr-x.pt") # Для кратного ускорения нужно экспортировать модель в TensorRT (файл export_trt_yolo.py) и использовать после rtdetr-x.engine

pose_checkpoint = "usyd-community/vitpose-plus-small"
pose_model = VitPoseForPoseEstimation.from_pretrained(
    pose_checkpoint, torch_dtype=dtype
).to(device)
pose_processor = AutoProcessor.from_pretrained(
    pose_checkpoint, use_fast=False
)


# У меня лично это нормально не работает, хз что не так
# compiled_pose_model = torch.compile(
#     pose_model, fullgraph=True, mode="reduce-overhead", dynamic=True
# )
#
# # Warm-up model
# dummy_batch_size = 1
# dummy_pixel_values = torch.zeros(
#     (dummy_batch_size, 3, pose_processor.size["height"], pose_processor.size["width"]),
#     dtype=dtype,
#     device=device,
# )
# dummy_inputs = {
#     "pixel_values": dummy_pixel_values,
#     "dataset_index": torch.zeros(dummy_batch_size, dtype=torch.int64, device=device),
# }
#
# for _ in range(10):
#     with torch.no_grad():
#         compiled_pose_model(**dummy_inputs)


# Tracker
buffer = BufferManager(max_history=max_history, device=device)


# ------------------------------
# Preprocessing functions
# ------------------------------
def preprocess_boxes(
    boxes_xyxy: np.ndarray, crop_height=256, crop_width=192, padding_factor=1.25
):
    aspect_ratio = crop_width / crop_height
    x_min, y_min, x_max, y_max = boxes_xyxy.T
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Match aspect ratio
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
    boxes = torch.from_numpy(boxes).float().to(device, non_blocking=True)

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
    min_value, max_value = -3, -1
    normalized_value = np.clip((value - min_value) / (max_value - min_value), 0, 1)
    red = int(normalized_value * 255)
    green = int((1 - normalized_value) * 255)
    return (0, red, green)  # BGR


def visualize_output(
    image: np.ndarray,
    boxes: torch.Tensor,
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    person_ids: np.ndarray,
    track2normal: dict,
    conf_threshold=0.01,
):
    mask = scores < conf_threshold
    keypoints[mask] = 0
    kp = sv.KeyPoints(xy=keypoints.cpu().numpy(), confidence=scores.cpu().numpy())
    detections = sv.Detections(
        xyxy=boxes.cpu().numpy(),
    )

    annotated = image.copy()

    edge_annotator = sv.EdgeAnnotator(color=sv.Color.YELLOW, thickness=1)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.ROBOFLOW, radius=2)

    annotated = edge_annotator.annotate(annotated, key_points=kp)
    annotated = vertex_annotator.annotate(annotated, key_points=kp)

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


def yolo_detect(frame):
    """Perform YOLO detection and return boxes, confs, class_ids, track_ids."""
    results = yolo_model.track(
        frame,
        verbose=False,
        classes=[0],
        persist=True,
        half=True,
    )[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()

    if results.boxes.id is not None:
        track_ids = results.boxes.id.cpu().numpy()
    else:
        track_ids = None
    return boxes, confs, class_ids, track_ids


def vitpose_infer(frame, boxes_xyxy, confs):
    """Perform pose inference on given frame and boxes."""
    if boxes_xyxy.shape[0] == 0:
        return None, None

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
        outputs = pose_model(**inputs)
    keypoints, scores = postprocess_keypoints(
        outputs.heatmaps, boxes_tensor, crop_height, crop_width
    )

    return (keypoints, scores, boxes_tensor)


def run_stg_inference(buffer_manager: BufferManager):
    """
    Build a tensor from the buffer and run STG_NF normality inference.
    """
    pose_tensor, conf_tensor, union_ids = buffer_manager.build_tensor()
    if pose_tensor is None:
        return None

    # Convert 17-keypoints to 18 (COCO style), then normalize
    kps_coco18 = keypoints17_to_coco18(pose_tensor)
    kps_norm = normalize_pose(kps_coco18)
    kps_final = kps_norm.permute(0, 3, 1, 2)

    scores = stg_inference.test_real_time(kps_final, conf_tensor)
    return scores, union_ids


# ------------------------------
# Asynchronous tasks
# ------------------------------


async def read_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        await frame_queue.put(frame)
    await frame_queue.put(None)  # signal end


async def detection_task(frame_queue, detection_queue):
    loop = asyncio.get_running_loop()
    while True:
        frame = await frame_queue.get()
        if frame is None:
            await detection_queue.put((None, None, None, None))
            break
        boxes, confs, class_ids, track_ids = await loop.run_in_executor(
            None, yolo_detect, frame
        )
        await detection_queue.put(
            (frame, (boxes, confs, class_ids, track_ids), None, None)
        )


async def pose_task(detection_queue, pose_queue):
    loop = asyncio.get_running_loop()
    while True:
        item = await detection_queue.get()
        frame, detection_data, _, _ = item
        if frame is None:
            await pose_queue.put((None, None, None, None))
            break

        boxes, confs, class_ids, track_ids = detection_data

        # Filter out non-person detections (class=0) with conf threshold
        mask = (class_ids == 0) & (confs > 0.01)
        if mask.sum() == 0:
            # Pass a frame with no detections
            await pose_queue.put((frame, None, None, None))
            continue

        valid_boxes = boxes[mask]
        valid_confs = confs[mask]
        valid_ids = track_ids[mask] if track_ids is not None else None

        # Pose inference
        keypoints_tuple = await loop.run_in_executor(
            None, vitpose_infer, frame, valid_boxes, valid_confs
        )
        # keypoints_tuple = vitpose_infer(frame, valid_boxes, valid_confs)
        await pose_queue.put((frame, keypoints_tuple, valid_ids, valid_confs))


# async def scoring_task(pose_queue, output_queue):
#     while True:
#         frame, keypoints_tuple, valid_ids, valid_confs = await pose_queue.get()
#         if frame is None:
#             await output_queue.put(None)
#             break
#
#         if keypoints_tuple is None or valid_ids is None or valid_confs is None:
#             # Nothing to do, just output the frame as-is
#             output_queue.put_nowait(frame)
#             continue
#
#         keypoints, scores, boxes_tensor = keypoints_tuple
#
#         # Update buffer with new data
#         detections_keypoints = torch.cat([keypoints, scores.unsqueeze(-1)], dim=-1)
#         tids_tensor = torch.from_numpy(valid_ids).long().to(device)
#         # Suppose detection confidence is in scores or a separate array
#         # but we have valid_confs... for demonstration, just set:
#         conf_tensor = torch.from_numpy(valid_confs).float().to(device)
#         buffer.update(tids_tensor, detections_keypoints, conf_tensor)
#
#         # If we have enough frames, run STG inference
#         frame_id = buffer.frame_counter
#         buffer.frame_counter += 1
#         annotated_frame = frame
#         if frame_id >= max_history - 1:
#             result = run_stg_inference(buffer)
#             if result is not None:
#                 normality_scores, union_ids = result
#                 track2normal = {}
#                 if union_ids.shape[0] == 1:
#                     track2normal[int(union_ids.item())] = float(normality_scores.item())
#                 else:
#                     for i, uid in enumerate(union_ids):
#                         track2normal[int(uid.item())] = float(
#                             normality_scores[i].item()
#                         )
#
#                 # Visualize
#                 annotated_frame = visualize_output(
#                     frame,
#                     boxes_tensor,
#                     keypoints,
#                     scores,
#                     valid_ids,
#                     track2normal,
#                 )
#
#         await output_queue.put(annotated_frame)


async def scoring_task(pose_queue, output_queue):
    global normality_history
    while True:
        frame, keypoints_tuple, valid_ids, valid_confs = await pose_queue.get()
        if frame is None:
            await output_queue.put(None)
            break
        if keypoints_tuple is None or valid_ids is None or valid_confs is None:
            val = normality_history[-1] if normality_history else -1
            normality_history.append(val)
            output_queue.put_nowait(frame)
            continue

        keypoints, scores, boxes_tensor = keypoints_tuple
        detections_keypoints = torch.cat([keypoints, scores.unsqueeze(-1)], dim=-1)
        tids_tensor = torch.from_numpy(valid_ids).long().to(device)
        conf_tensor = torch.from_numpy(valid_confs).float().to(device)
        buffer.update(tids_tensor, detections_keypoints, conf_tensor)

        frame_id = buffer.frame_counter
        buffer.frame_counter += 1
        annotated_frame = frame
        if frame_id >= max_history - 1:
            result = run_stg_inference(buffer)
            if result is not None:
                normality_scores, union_ids = result
                if normality_scores.size == 1:
                    min_val = normality_scores.item()
                else:
                    min_val = np.min(normality_scores)
                normality_history.append(min_val)
                track2normal = {}
                if union_ids.size(dim=0) == 1:
                    track2normal[int(union_ids.item())] = float(normality_scores.item())
                else:
                    for i, uid in enumerate(union_ids):
                        track2normal[int(uid)] = float(normality_scores[i])
                annotated_frame = visualize_output(
                    frame, boxes_tensor, keypoints, scores, valid_ids, track2normal
                )
            else:
                val = normality_history[-1] if normality_history else -1
                normality_history.append(val)
        else:
            val = normality_history[-1] if normality_history else -1
            normality_history.append(val)

        await output_queue.put(annotated_frame)


def create_graph_image(history, width, height):
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.2)
    draw_width = width - 2 * margin_x
    draw_height = height - 2 * margin_y
    graph_img = np.full((height, width, 3), 255, dtype=np.uint8)

    if len(history) == 0:
        return graph_img

    # Use the last min(len(history), draw_width) points for horizontal scaling
    n_points = min(len(history), draw_width)
    data = history[-n_points:]

    # Dynamically compute data range and add padding for better visibility
    data_min = min(data)
    data_max = max(data)
    if data_max == data_min:
        data_max += 1e-6
    padding = 0.1 * (data_max - data_min)
    data_min -= padding
    data_max += padding

    pts = []
    for i, value in enumerate(data):
        x = (
            margin_x + int((i / (n_points - 1)) * draw_width)
            if n_points > 1
            else margin_x + draw_width // 2
        )
        norm_val = (value - data_min) / (data_max - data_min)
        y = margin_y + int((1 - norm_val) * draw_height)
        pts.append((x, y))
    if len(pts) > 1:
        pts_np = np.array(pts, np.int32)
        cv2.polylines(
            graph_img,
            [pts_np],
            isClosed=False,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Draw vertical (y) and horizontal (x) axes
    cv2.line(
        graph_img,
        (margin_x, margin_y),
        (margin_x, margin_y + draw_height),
        (0, 0, 0),
        1,
    )
    cv2.line(
        graph_img,
        (margin_x, margin_y + draw_height),
        (margin_x + draw_width, margin_y + draw_height),
        (0, 0, 0),
        1,
    )

    # Add ticks and labels on vertical axis (normality)
    cv2.putText(
        graph_img,
        f"{data_max:.2f}",
        (2, margin_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        graph_img,
        f"{data_min:.2f}",
        (2, margin_y + draw_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )

    # Add ticks and labels on horizontal axis (frames)
    cv2.putText(
        graph_img,
        "0",
        (margin_x, margin_y + draw_height + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        graph_img,
        f"frame: {n_points - 1}",
        (margin_x + draw_width - 20, margin_y + draw_height + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )

    # Add axis titles
    cv2.putText(
        graph_img,
        "frames",
        (margin_x + draw_width // 2 - 20, margin_y + draw_height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        graph_img,
        "normality",
        (5, margin_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    return graph_img


normality_history = []


async def writer_task(
    output_queue,
    display=True,
    output_file=None,
    video_fps=0,
    frame_width=0,
    frame_height=0,
):
    writer = None
    graph_height = 200  # высота области для графика
    if output_file is not None and frame_width > 0 and frame_height > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Итоговый кадр = исходный кадр + график (вертикальное объединение)
        writer = cv2.VideoWriter(
            output_file, fourcc, video_fps, (frame_width, frame_height + graph_height)
        )
    prev_time = time.time()
    while True:
        frame = await output_queue.get()
        if frame is None:
            break

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        fps_value = 1.0 / dt if dt > 0 else 0.0
        frame_time_ms = dt * 1000.0
        if video_fps > 0:
            realtime_interval = 1.0 / video_fps
            realtime_status = "ahead" if dt < realtime_interval else "slower"
        else:
            realtime_status = "unknown"

        overlay_text = [
            f"FPS: {fps_value:.2f}",
            f"Frame time: {frame_time_ms:.1f} ms",
            f"Realtime: {realtime_status}",
        ]
        y0 = 30
        dy = 30
        for i, line in enumerate(overlay_text):
            y = y0 + i * dy
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        # Создаём изображение графика с динамическим масштабированием и разметкой
        graph_img = create_graph_image(normality_history, frame.shape[1], graph_height)
        combined = np.vstack([frame, graph_img])

        if writer is not None:
            writer.write(combined)
        if display:
            cv2.imshow("Output", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


# ------------------------------
# Main async pipeline
# ------------------------------
async def async_main(input_source, output_path=None):
    cap = cv2.VideoCapture(input_source)
    fps_capture = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    frame_queue = asyncio.Queue()
    detection_queue = asyncio.Queue()
    pose_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    tasks = [
        asyncio.create_task(read_frames(cap, frame_queue)),
        asyncio.create_task(detection_task(frame_queue, detection_queue)),
        asyncio.create_task(pose_task(detection_queue, pose_queue)),
        asyncio.create_task(scoring_task(pose_queue, output_queue)),
        asyncio.create_task(
            writer_task(
                output_queue,
                display=True,
                output_file="stg_test.avi",
                video_fps=int(fps_capture),
                frame_width=frame_width,
                frame_height=frame_height,
            )
        ),
    ]

    await asyncio.gather(*tasks)
    cap.release()


def main():
    video_path = r"test_video.mp4"

    start = time.time()
    asyncio.run(async_main(video_path))
    torch.cuda.synchronize()
    print("Time taken: ", time.time() - start)

    # Для прохода по всей тестовой части датасета
    # path = r"F:\shanghaitech\testing\videos"
    # file_names = os.listdir(path)
    # for file_name in file_names:
    #     buffer.frame_counter = 0
    #     global normality_history
    #     normality_history = []
    #     if yolo_model.predictor is not None:
    #         yolo_model.predictor.trackers[0].reset()
    #     asyncio.run(async_main(os.path.join(path, file_name)))


if __name__ == "__main__":
    main()
