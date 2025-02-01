Запуск

Запускаем generate_data.py на своем любом видео с толпой людей. Скрипт создаст `08_0044_alphapose_tracked_person.json` (временное название, необходимо чтобы STG-NF корректно подгрузил этот файл) файл со всем необходимым чтобы запустить на этом STG-NF
Далее нужен весь работающий проект STG-NF с чекпоинтами и структурой папок под датасеты(но сами файлы не нужны!)
Переносим файл `08_0044_alphapose_tracked_person.json` по такому пути
![image](https://github.com/user-attachments/assets/49de67c7-9686-4b58-bb5f-7074a25bd266)
Заменяем файл `train_eval.py` в STG-NF на файл из этого проекта
Запускаем такую команду:
`python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar`
После этого нам сгенерируется файл `normality_scores.json` с аномалиями.
Далее запускаем `visualization.py` (в нем должен быть прописан путь до файла `normality_scores.json` и до файла видео .mp4)
