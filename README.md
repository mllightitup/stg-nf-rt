## Установка
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --upgrade`
- `pip install opencv-python ultralytics supervision transformers lap onnx>=1.12.0 onnxslim onnxruntime-gpu tensorrt --upgrade`

## Экспорт модели TensorRT

Чтобы сильно ускорить модель детектора(RTDETR или YOLO) можно её экспортировать с помощью `export_trt_yolo.py`.

После успешного экспорта в папке `detector_weights` появится модель `rtdetr-x.engine"`

В файлах `pipeline_async.py` | `pipeline_sync.py` нужно поменять свою строку на такую: 

`yolo_model = RTDETR(r"detector_weights/rtdetr-x.engine")`

## Результаты обучения STG-NF
Датасет ShanghaiTech был размечен с помощью `pipeline_async.py` (с отключенным модулем STG-NF и с дополнительным модулем для разметки) со средней скоростью >24fps, то есть быстрее реал тайма. Точность можно сделать на несколько процентов выше, но в таком случае в realtime мы уже не укладываемся.

Были размечены полность train/test сеты:
- train - 330 видеороликов (274 515 кадров)
- test - 107 видеороликов (42 883 кадров)

[Тут добавить ссылку на разметку]

PC specs:
- GPU: NVIDIA RTX 4070 12GB
- CPU: AMD Ryzen 5 7500F


На датасете **ShanghaiTech**:

![image](https://github.com/user-attachments/assets/5fa828b2-70de-4ee1-a937-2ce15d17fa6d)

На датасете **ShanghaiTech-HR** (human related):

![image](https://github.com/user-attachments/assets/febd8787-bc93-4056-b5be-7c3feb93a651)
