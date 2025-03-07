### Установка и зависимости

Установка под чистый/пустой virtualenv для Windows 10/11 и python 3.12.9

Сначала обновляем/устанавливани базовые библиотеки
- `python3 -m pip install --upgrade pip`
- `python3 -m pip install wheel`?
- `pip install setuptools packaging --upgrade`

На всякий случай удаляем кэш предыдущих установок
- `pip cache remove "tensorrt*"`

Замените путь до установки TensorRT.
Эти библиотеки лежат по пути(пример):

`C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.8.0.43\python`

- `pip install --upgrade "ВАШ_ПУТЬ\tensorrt-10.8.0.43-cp312-none-win_amd64.whl" ВАШ_ПУТЬ\tensorrt_lean-10.8.0.43-cp312-none-win_amd64.whl ВАШ_ПУТЬ\tensorrt_dispatch-10.8.0.43-cp312-none-win_amd64.whl`

Код тестировался на  `torch-2.6.0+cu126` `torchaudio-2.6.0+cu126` `torchvision-0.21.0+cu126`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

`pip install git+https://github.com/mikel-brostrom/boxmot.git git+https://github.com/mikel-brostrom/ultralytics.git --no-deps`

`pip install tqdm transformers supervision opencv-python scipy scikit-learn matplotlib loguru pandas gdown ftfy lap filterpy thop psutil --upgrade`