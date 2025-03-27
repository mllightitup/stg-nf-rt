## TODO
- Refactor(аргументы для запуска скиптов --input --output --use-trt-detector --use-trt-pose ...)
- Cleanup STG-NF
- STG-NF compile
- Найти что не так в `pipeline_sync.py` и почему он немного медленее
- Унифицировать `pipeline_sync.py` и `pipeline_async.py`
## Установка

- Создайте проект с .venv любым удобным способом [**python >=3.12 и <3.13**]
```bash
git clone -b new https://github.com/mllightitup/stg-nf-rt.git
```
```bash
pip install uv
```
```bash
uv pip install . или uv sync
```


## Экспорт модели TensorRT
Чтобы сильно ускорить модели **RTDETR** и **VITPOSE** экспортируйте их с помощью **TensorRT**.
### RTDETR
Запустите `export_trt_yolo.py`

После успешного экспорта в папке `detector_weights` появится модель `rtdetr-x.engine`

В файлах `pipeline_async.py` | `pipeline_sync.py` нужно поменять `.pt` на `.engine`: 

`yolo_model = RTDETR(r"detector_weights/rtdetr-x.engine")`

### VITPOSE
Запустите `export_trt_vitpose.py`

После успешного экспорта в папке `pose_weights` появится модель `vitpose-plus-small.ep`

В файлах `pipeline_async.py` | `pipeline_sync.py` нужно поменять... [TODO]

## Результаты обучения STG-NF
Датасет ShanghaiTech был размечен с помощью `pipeline_async.py` (с отключенным модулем STG-NF и с дополнительным модулем для разметки) со средней скоростью >24fps, то есть быстрее реал тайма. Точность можно сделать на несколько процентов выше, но в таком случае в realtime мы уже не укладываемся.

Были размечены полность train/test сеты:
- train - 330 видеороликов (274 515 кадров)
- test - 107 видеороликов (42 883 кадров)

[TODO: тут добавить ссылку на разметку]

PC specs:
- GPU: NVIDIA RTX 4070 12GB
- CPU: AMD Ryzen 5 7500F


На датасете **ShanghaiTech**:

![image](https://github.com/user-attachments/assets/5fa828b2-70de-4ee1-a937-2ce15d17fa6d)

На датасете **ShanghaiTech-HR** (human related):

![image](https://github.com/user-attachments/assets/febd8787-bc93-4056-b5be-7c3feb93a651)

## Визуализация

![stg_test (1)](https://github.com/user-attachments/assets/10eb2b88-5c29-4d60-90f0-d3c546cb465c)




