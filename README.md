# Wavefront Reconstruction

Восстановление двумерного волнового фронта `U(x, y)` по градиентам `dU/dx`,
`dU/dy` на разреженной сетке датчиков. Проект перенесён из Jupyter Notebook
в устанавливаемый Python-пакет; Colab и запуск ячеек ноутбука не требуются.

Поддерживаются:

- **DeepONet**: градиенты датчиков → волновой фронт;
- **FNO**: градиенты регулярной сетки → волновой фронт; в режиме `sensor`
  вход сначала интерполируется с датчиков;
- **двухстадийная модель**: обучение DeepONet для карты градиентов,
  затем совместное обучение с FNO, инициализированным случайно;
- **трёхстадийная модель**: обучение DeepONet, обучение FNO по его выходам,
  затем совместное дообучение;
- **Poisson baseline** и последовательное сравнение всех моделей.

## Установка

Нужен **Python 3.11 или 3.12**, рекомендуется 3.11. Базовая установка работает
на CPU и не требует CUDA. Проверки выполнялись на Linux / Python 3.11.
Для Windows рекомендуется WSL2 с Ubuntu; macOS и нативная Windows в этой
среде не проверялись. Установку GPU-версии JAX нужно согласовать отдельно
с CUDA/драйвером и закреплённой здесь версией JAX 0.4.26.

```bash
git clone https://github.com/ZakirovGT/Wavefront.git
cd Wavefront
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Если проект скачан ZIP-архивом, распакуйте его и выполните те же команды начиная
с создания `.venv` внутри каталога проекта. В PowerShell активация venv:
`.venv\Scripts\Activate.ps1`. Все команды ниже выполняются **из корня репозитория**.
Зависимости объявлены в `pyproject.toml`, транзитивные версии зафиксированы
в `constraints.txt`. Можно также установить пакет командой `pip install .`;
для повторения проверенной среды используйте `requirements.txt`.

## Быстрый запуск на CPU

В `configs/smoke/` находятся маленькие конфигурации: сетка 8×8, 6 обучающих
и 3 тестовых примера, по 2 шага обучения. Это проверка работоспособности,
а не обучение модели с полезной точностью. Первая JIT-компиляция занимает время.

```bash
python scripts/generate_dataset.py --config configs/smoke/standalone_data.yaml --output-dir datasets/smoke
python scripts/train_fno.py --config-dir configs/smoke --dataset-dir datasets/smoke --results-root results/smoke
python scripts/train_deeponet.py --config-dir configs/smoke --dataset-dir datasets/smoke --results-root results/smoke
python scripts/train_two_stage.py --config configs/smoke/two_stage.yaml --dataset-dir datasets/smoke
python scripts/train_three_stage.py --config configs/smoke/three_stage.yaml --dataset-dir datasets/smoke
```

Каждая команда обучения печатает `Run directory`. Сохраните этот путь для
предсказаний и сравнения. Повторно используйте готовый dataset; генератор
защищает существующие данные от перезаписи. Для намеренной замены есть
`--overwrite`. В среде без дисплея можно установить `MPLBACKEND=Agg`.

## Полноразмерные эксперименты

Конфигурации непосредственно в `configs/` задают исследовательские размеры:
сетка 24×24, 10 000 обучающих и 1 000 тестовых примеров, десятки тысяч шагов.
Перед длительным запуском подберите размер batch под доступную память.
Обученных весов в репозитории нет.

```bash
python scripts/generate_dataset.py --config configs/standalone_data.yaml --output-dir datasets/experiment
python scripts/train_deeponet.py --dataset-dir datasets/experiment
python scripts/train_fno.py --dataset-dir datasets/experiment
python scripts/train_two_stage.py --dataset-dir datasets/experiment
python scripts/train_three_stage.py --dataset-dir datasets/experiment
```

Без `--dataset-dir` модели генерируют данные по своим конфигурациям. В многостадийных
моделях при этом используются отдельные seeds; с `--dataset-dir` все стадии
используют один сохранённый train/test split. `--steps` у самостоятельных
моделей переопределяет число шагов; для короткой проверки используйте именно
`configs/smoke`, поскольку один только `--steps` не уменьшает модель и dataset.

Семейства синтетических данных: Zernike, spiral и фильтрованные случайные
искажения. Их доли задаются `frac_zernike`, `frac_spiral`, `frac_distortion`
(сумма равна 1). Генерация нормализует фронт каждого примера и тем же масштабом
делит градиенты. Параметры шума и seed находятся в YAML.

## Датчики и предсказания из CSV

Включён исходный **`1.csv` — 483 датчика** с колонками `N,X,Y,Zx,Zy,Ampl,Flag(0/1)`.
Обучение использует `X,Y` в пикселях и преобразование из
`wavefront.data.sensors.sensor_affine_transform`; порядок строк сохраняется.
Там находятся калибровочные коэффициенты именно для этой геометрии.
Для другой установки сначала настройте преобразование и переобучите модель.

Замените `RUN_DIR` путём, напечатанным командой обучения FNO:

```bash
python scripts/predict_fno.py --run-dir RUN_DIR --measurements 1.csv --sensor-layout 1.csv --gx-column Zx --gy-column Zy --sensor-id-column N --layout-sensor-id-column N --normalization-scale 1 --output-dir results/prediction
```

Для остальных моделей используйте `predict_deeponet.py`, `predict_two_stage.py`
или `predict_three_stage.py` с соответствующим каталогом обучения. CLI
DeepONet работает с моделью, обученной в режиме `sensor`.

- `--layout-coordinate-system pixel` — по умолчанию, соответствует `1.csv`.
  Для уже нормализованных `X,Y` передайте `normalized`.
- В собственном CSV по умолчанию ожидаются `gx,gy`. Для нескольких примеров
  добавьте колонку идентификатора и `--sample-id-column sample_id`.
- Идентификаторы датчиков позволяют переставить измерения в порядок layout.
  Сам layout должен соответствовать порядку и геометрии обучения.
- Все датчики участвуют в обучении. Флаг `Flag(0/1)` автоматически не фильтруется;
  для проверки измерений передайте `--valid-column 'Flag(0/1)'`.
  Отсутствующие/невалидные измерения по умолчанию вызывают ошибку.

`--normalization-scale` делит входные градиенты и умножает выходной фронт;
он должен соответствовать физической калибровке измерений и обучению.
Значение `1` выше проверяет интерфейс на примере CSV. Оно не устанавливает
физическую калибровку `Zx,Zy`: градиенты должны быть приведены к производным
по нормализованным координатам модели. Постоянную составляющую фронта нельзя
определить по градиентам; её задаёт `--wavefront-offset` (по умолчанию 0).

Выход содержит `wavefront_predictions_normalized.npy`,
`wavefront_predictions_physical.npy`, таблицы CSV, PNG и `inference_manifest.json`.
Параметр `--units` записывает обозначение единиц в метаданные.

## Сравнение моделей

В копии `configs/benchmark.yaml` (или `configs/smoke/benchmark.yaml` для малой сетки)
укажите реальные каталоги обучения в секции `runs`:

```yaml
runs:
  deeponet: results/smoke/deeponet/ВАШ_ЗАПУСК
  fno: results/smoke/fno/ВАШ_ЗАПУСК
  two_stage: results/smoke/two_stage/ВАШ_ЗАПУСК
  three_stage: results/smoke/three_stage/ВАШ_ЗАПУСК
```

`null` пропускает модель; Poisson вычисляется всегда. Размер сетки и геометрия
должны соответствовать обучению. Для smoke-конфигурации:

```bash
python scripts/run_benchmark.py --config configs/smoke/benchmark.yaml --dataset-dir datasets/smoke --fail-fast
```

Используйте конфигурацию, в которую записали свои пути. Команда создаёт
`report.json`, метрики relative L2, замеры времени и сравнения в PNG.
В режиме `sensor` FNO и Poisson получают интерполяцию измерений датчиков;
в режиме `grid` — готовые градиенты регулярной сетки.

## Тесты и перенос результатов

```bash
python -m pip install -c constraints.txt '.[dev]'
python -m pytest -q
```

Тесты проверяют CLI, генерацию всех трёх семейств данных, обучение четырёх моделей,
checkpoint restore, перенос каталогов, CSV-предсказания, общий benchmark,
визуализации, dropout и отключение промежуточных checkpoints. Они работают
на CPU; для полного набора нужно несколько минут. CI настроен на Python 3.11/3.12.

Для переноса dataset копируйте весь каталог с `arrays.npz`, `generation_config.yaml`,
`metadata.json` и `sensor_layout.csv`. Для переноса модели копируйте **весь каталог
запуска**, включая stage-каталоги для многостадийных моделей. Координаты датчиков
сохранены вместе с параметрами. Восстановление размещает параметры на устройстве
нового компьютера, не требуя исходного пути dataset или исходного GPU.
При CSV-предсказании по-прежнему указывайте соответствующий sensor layout.

## Структура

```text
configs/               # полные конфигурации и smoke-примеры
scripts/               # генерация, обучение, предсказания, benchmark
src/wavefront/
  physics/             # сетки, Zernike, spiral, градиенты, фильтры
  data/                # датчики, генераторы, dataset artifacts
  models/              # DeepONet и FNO
  training/            # обучение самостоятельных моделей
  metamodel/           # карта градиентов и совместное обучение
  pipelines/           # двух- и трёхстадийные сценарии
  inference/           # загрузка и предсказания из CSV
  evaluation/          # метрики, benchmark, визуализации
tests/                 # CPU-регрессии
1.csv                  # исходный layout и пример измерений
```

Прежние иллюстрации из ноутбука сохранены в `images/sensor_s1.png` и
`images/regular_z1.png`. Их точность не является результатом коротких smoke-тестов.
Длительное обучение и GPU-производительность требуют отдельной проверки.
