## Установка
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --upgrade`
- `pip install opencv-python ultralytics supervision transformers lap onnx>=1.12.0 onnxslim onnxruntime-gpu tensorrt --upgrade`

## Экспорт модели TensorRT

Чтобы сильно ускорить модель детектора(RTDETR или YOLO) можно её экспортировать с помощью `export_trt_yolo.py`.

После успешного экспорта в папке `detector_weights` появится модель `rtdetr-x.engine"`

В файлах `pipeline_async.py` | `pipeline_sync.py` нужно поменять свою строку на такую: 

`yolo_model = RTDETR(r"detector_weights/rtdetr-x.engine")`

## Результаты обучения STG-NF
На датасете **ShanghaiTech**:

![image](https://github.com/user-attachments/assets/5fa828b2-70de-4ee1-a937-2ce15d17fa6d)

На датасете **ShanghaiTech-HR** (human related):

![image](https://github.com/user-attachments/assets/febd8787-bc93-4056-b5be-7c3feb93a651)
