import torch
import torch_tensorrt
from transformers import VitPoseForPoseEstimation

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

# Сохраняем скомпилированную модель в формате TensorRT
torch_tensorrt.save(trt_model, "pose_weights/vitpose-plus-small.ep")
