# import torch
# import requests
# from PIL import Image
# from transformers import (
#     AutoProcessor,
#     RTDetrForObjectDetection,
#     VitPoseForPoseEstimation,
# )
# import torch_tensorrt
#
# print(torch_tensorrt.dtype)
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Этап 1: Детектор людей
# person_image_processor = AutoProcessor.from_pretrained(
#     "PekingU/rtdetr_r50vd_coco_o365", use_fast=True
# )
# person_model = RTDetrForObjectDetection.from_pretrained(
#     "PekingU/rtdetr_r50vd_coco_o365"
# )
# person_model = person_model.to(device)
#
# url = "http://images.cocodataset.org/val2017/000000000139.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# inputs_person = person_image_processor(images=image, return_tensors="pt").to(device)
# with torch.no_grad():
#     outputs_person = person_model(**inputs_person)
#
# results = person_image_processor.post_process_object_detection(
#     outputs_person,
#     target_sizes=torch.tensor([(image.height, image.width)]),
#     threshold=0.3,
# )
# result = results[0]
# person_boxes = result["boxes"][result["labels"] == 0].cpu().numpy()
# # Преобразуем координаты из (x1, y1, x2, y2) в (x, y, w, h)
# person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
# person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
#
# # Этап 2: Определение поз
# image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-small")
# model = VitPoseForPoseEstimation.from_pretrained(
#     "usyd-community/vitpose-plus-small", torch_dtype=torch.float16
# )
# model = model.to(device)
#
# # Подготовка входов
# inputs_pose = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(
#     device
# )
# sample_pixel_values = inputs_pose["pixel_values"].half()
#
# # Фиксируем boxes и dataset_index, так как они не меняются для данной картинки
# fixed_boxes = [torch.tensor(person_boxes, device=device)]
# dataset_index = torch.tensor([0], device=device)
#
#
# # Обёртка без передачи boxes и dataset_index через forward
# class VitPoseWrapper(torch.nn.Module):
#     def __init__(self, model, dataset_index):
#         super(VitPoseWrapper, self).__init__()
#         self.model = model
#         self.register_buffer("dataset_index", dataset_index)
#
#     def forward(self, pixel_values):
#         # Передаем только pixel_values и dataset_index, без боксов!
#         return self.model(pixel_values=pixel_values, dataset_index=self.dataset_index)
#
#
# vitpose_wrapper = VitPoseWrapper(model, dataset_index)
#
# # Компиляция модели в TensorRT FP16 (теперь вход всего один: pixel_values)
# trt_model = torch_tensorrt.compile(
#     vitpose_wrapper,
#     # inputs=[torch_tensorrt.Input(sample_pixel_values.shape)],
#     inputs=torch_tensorrt.Input(
#         min_shape=[1, 3, 256, 192],
#         opt_shape=[5, 3, 256, 192],
#         max_shape=[20, 3, 256, 192],
#         dtype=torch.float16,
#     ),
#     enabled_precisions={torch.half},
#     ir="dynamo",
# )
# torch_tensorrt.save(trt_model, "trt1.ep")
# model = torch.export.load("trt1.ep").module()
# print(model(sample_pixel_values))
# print(sample_pixel_values.shape)
# # trt_model
#
# # Проведение инференса с TensorRT-моделью
# with torch.no_grad():
#     outputs_trt = trt_model(sample_pixel_values)
#     # print(outputs_trt)
# # Постобработка результатов
# pose_results = image_processor.post_process_pose_estimation(
#     outputs_trt, boxes=[person_boxes]
# )
# image_pose_result = pose_results[0]
# print(image_pose_result)


import torch
import torch_tensorrt
from transformers import VitPoseForPoseEstimation

# Устанавливаем устройство: GPU (cuda) или CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель VitPose в режиме FP16
model = (
    VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-plus-small", torch_dtype=torch.float16
    )
    .eval()
    .to(device)
)


# Определяем обёртку, фиксирующую dataset_index в качестве буфера.
# Модель ожидает на вход словарь с ключами "pixel_values" и "dataset_index",
# поэтому фиксируем dataset_index, передаваемый в forward.
class VitPoseWrapper(torch.nn.Module):
    def __init__(self, model, dataset_index):
        super(VitPoseWrapper, self).__init__()
        self.model = model
        self.register_buffer("dataset_index", dataset_index)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values, dataset_index=self.dataset_index)


# Фиксируем dataset_index (например, для батча размера 1)
dataset_index = torch.tensor([0], device=device)
wrapped_model = VitPoseWrapper(model, dataset_index)

# Подготавливаем пример входного тензора (формат [batch, channels, height, width])
# Здесь выбран размер 1x3x256x192, приводим к FP16 для соответствия модели.
sample_input = torch.randn(1, 3, 256, 192, device=device).half()

# Компилируем модель в формат TensorRT (FP16) с динамическими размерами батча
trt_model = torch_tensorrt.compile(
    wrapped_model,
    inputs=torch_tensorrt.Input(
        min_shape=[1, 3, 256, 192],
        opt_shape=[6, 3, 256, 192],
        max_shape=[30, 3, 256, 192],
        dtype=torch.float16,
    ),
    enabled_precisions={torch.half},
    ir="dynamo",
)

# Создаём директорию для сохранения, если её ещё нет
# os.makedirs("detector_weights", exist_ok=True)

# Сохраняем скомпилированную модель в формате TensorRT
torch_tensorrt.save(trt_model, "detector_weights/vitpose-plus-small.ep")
