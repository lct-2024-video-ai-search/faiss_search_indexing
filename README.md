# Сервис индексирования и поиска
Документация для сервиса, построенного на Faiss для индексирования и быстрого поиска похожих видео по запросу.

Сервис предоставляет два основных эндпойнта:

1. /create_video_index (POST)
   - Описание: Создание или обновление индекса для видео.
   - Входные данные:
     - VideoDescription: Описание видео.
     - VideoMovementDesc: Описание движений в видео.
     - VideoSpeechDescription: Описание речи в видео.
     - Index: Индекс видео в ClickHouse.
   - Выходные данные:
     - status: Статус выполнения операции ("Success").

2. /search (GET)
   - Описание: Поиск похожих видео по запросу.
   - Входные данные:
     - query: Текстовый запрос.
   - Выходные данные:
     - ids: Список идентификаторов (индексов) видео, упорядоченных по релевантности запросу.

# Внутренние функции:

1. create_index_index_videos()
   - Описание: Создание индекса для всех видео в базе данных.
   - Вход: Нет.
   - Выход: Возвращает индекс Faiss и список идентификаторов видео.

2. index_video(video_index, video_description, video_movement_desc, video_speech_description, index, index_ids)
   - Описание: Добавление нового видео в индекс.
   - Вход:
     - video_index: Индекс видео в ClickHouse.
     - video_description: Описание видео.
     - video_movement_desc: Описание движений в видео.
     - video_speech_description: Описание речи в видео.
     - index, index_ids: Индекс faiss, общий список id видео из ClickHouse
   - Выход: Статус “Успех”, означающий успешно обновленный индекс Faiss.

3. search(query)
   - Описание: Поиск похожих видео по запросу.
   - Вход:
     - query: Текстовый запрос.
   - Выход:
     - ids: Список идентификаторов найденных видео.

# Общая архитектура сервиса:
- Используется библиотека Faiss для быстрого поиска похожих видео.
- Для извлечения векторных представлений пользовательских запросов и описания видео используется предобученная модель BGE-M3.
- Сервис реализован с помощью FastAPI.
- Для запуска сервиса в production-среде рекомендуется использовать Docker.

# Примеры использования:

1. Добавление нового видео в индекс:
```
indexInfo = IndexInfo(
    VideoDescription="Видео о природе",
    VideoMovementDesc="Плавные движения камеры",
    VideoSpeechDescription="Отсутствует речь",
    Index=10
)
response = app.create_video_index(indexInfo)
print(response.status)  
```
Выведет "Success"

2. Поиск похожих видео по запросу:
```
query = "Красивые пейзажи"
response = app.search(query)
print(response.ids)  
````
Выведет список индексов похожих видео

# Локальный запуск:

```
pip install -r requirements.txt
```

```
uvicorn main:app --host 0.0.0.0 --port 8000
```
