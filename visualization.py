import json
import cv2
import numpy as np

# Параметры
json_path = "08_0044_alphapose_tracked_person.json"
value_json_path = "normality_scores.json"
video_path = "crowd.mp4"
output_video_path = "crowd_output.mp4"

# Загрузка данных
with open(json_path, "r") as file:
    data = json.load(file)

with open(value_json_path, "r") as file:
    values_data = json.load(file)

# Загрузка видео
cap = cv2.VideoCapture(video_path)

# Проверяем, открыто ли видео
if not cap.isOpened():
    raise Exception("Не удалось открыть видео.")

# Получение параметров видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Настройка видео для записи
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Обработка каждого кадра
frame_idx = 0
delay = int(500 / fps)  # Задержка для отображения


# Функция для преобразования значения value в цвет
def value_to_color(value):
    # Нормализуем значение в диапазон [0, 1]
    min_value, max_value = (
        -15,
        0,
    )  # Например, предположим, что минимальные и максимальные значения value в этом диапазоне
    normalized_value = np.clip((value - min_value) / (max_value - min_value), 0, 1)

    # Используем normalized_value для создания цвета: красный - зеленый
    red = int(normalized_value * 255)  # Чем меньше значение, тем краснее
    green = int((1 - normalized_value) * 255)  # Чем больше значение, тем зеленее
    return 0, red, green  # Цвет в формате (B, G, R)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Проверяем наличие данных для текущего кадра
    for person_id, frames in data.items():
        if str(frame_idx) in frames:
            frame_data = frames[str(frame_idx)]

            # Данные бокса
            box = frame_data["boxes"]
            confidence = frame_data["scores"]
            center_x, center_y, w, h = map(int, box)

            # Преобразуем координаты центра в верхний левый угол
            top_left_x = int(center_x - w / 2)
            top_left_y = int(center_y - h / 2)

            # Рисуем бокс

            # # Подпись ID персоны с обводкой
            # cv2.putText(
            #     frame,
            #     f"Person {person_id}",
            #     (top_left_x, top_left_y - 15),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 0, 0),  # Черный цвет (обводка)
            #     2,
            # )
            # cv2.putText(
            #     frame,
            #     f"Person {person_id}",
            #     (top_left_x, top_left_y - 15),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),  # Зеленый цвет
            #     1,
            # )

            # # Подпись confidence с обводкой
            # cv2.putText(
            #     frame,
            #     f"Conf: {confidence:.2f}",
            #     (top_left_x, top_left_y - 35),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 0, 0),  # Черный цвет (обводка)
            #     2,
            # )
            # cv2.putText(
            #     frame,
            #     f"Conf: {confidence:.2f}",
            #     (top_left_x, top_left_y - 35),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 255),  # Желтый цвет
            #     1,
            # )

            # Проверяем наличие значения в values_data
            value = "None"
            if person_id in values_data and str(frame_idx) in values_data[person_id]:
                value = values_data[person_id][str(frame_idx)]

            # Преобразуем значение в цвет
            if value != "None":
                color = value_to_color(value)
                cv2.rectangle(
                    frame,
                    (top_left_x, top_left_y),
                    (top_left_x + w, top_left_y + h),
                    color,
                    1,
                )
                # Подпись значения с обводкой и цветом
                cv2.putText(
                    frame,
                    f"{value:.2f}",
                    (top_left_x, top_left_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Черный цвет (обводка)
                    3,
                )
                cv2.putText(
                    frame,
                    f"{value:.2f}",
                    (top_left_x, top_left_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,  # Цвет в зависимости от значения
                    1,
                )

    out.write(frame)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):  # Учитываем задержку
        break

    frame_idx += 1

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()

print("Обработанное видео сохранено в", output_video_path)
