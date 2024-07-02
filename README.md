# Credit Risk


Язык: **Python**

Системные требования
------
 - Python >=3.8 (проверено на 3.8.10 и 3.10.12)
 - Библиотеки: numpy, pandas, fastparquet, matplotlib, dill,
scikit-learn, catboost, xgboost, lightgbm (см. `requirements.txt`)
 - ОЗУ: 16 MB + 8 MB swap

Подготовка  данных
------
Скопируйте `train_data` и `train_target.csv` в папку `data`

Далее для получения предобработанных аггрегированных данных:
```bash
python3 bin/data_loader.py
```
Время: 100 минут

Данные запишутся в папку `data/aggregated`

Время: 10 минут на CPU

Тюнинг модели и валидация
------
Осуществляется редактированием и запуском скрипта `bin/training.py` (логгируется)

Обучение модели 
------
```bash
python3 bin/training.py [--name \<model name\>] [--n-proc \<n\>] [--gpu]
```

По умолчанию `\<model name\>=catboost`

При использовании `--gpu`, рекомендуется указать `--n-proc 1`, иначе может не хватить памяти

Обученная модель сохраняется в dill-pkl файл в папку `models`

Время: 1.5 минуты на GPU, 25 минут на CPU

Создание пайплайна из уже обученной модели с предобработкой
------
Без проверки на тестовых данных
```bash
python3 bin/final_pipeline.py [--name \<model name or path\>] [--n-proc \<n\>] [--gpu]
```

С проверкой на тестовых данных
```bash
python3 bin/final_pipeline.py [--name \<model name or path\>] [--n-proc \<n\>] [--gpu] --test-file data/train_data/train_data_0.pq
```
или с другим файлом

По умолчанию `\<model name\>=catboost`

Опции `--gpu` и `--n-proc` определяют поведение модели при использовании

Пайплайн сохраняется в dill-pkl файл

Проверка пайплайна и получение предсказаний
------
```bash
python3 bin/check_pipeline.py [--name \<model name or path\>] --input data/train_data/train_data_0.pq --output data/prediction_0.pq
```

По умолчанию `\<model name\>=catboost`

Если нужно получить предсказание по нескольким файлам, то надо запускать по очереди, чтобы хватило памяти.

Время: 10 минут

