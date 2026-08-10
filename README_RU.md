# CayleyPy 4×4×4: полный write-up решения 46 296

## 0. Итог

Финальный локально проверенный submission:

- файл: `cube4_submission_46296.csv`;
- задач: **1 043** (`initial_state_id` от `0` до `1042`);
- score: **46 296**;
- средняя длина: **44.387344** хода;
- exact replay: **1 043/1 043 solved**;
- SHA-256: `8a10867da63d388de0b0131be38badddd9b4c0fa7edfe0906f188a6b9cbd21f7`;
- исходный пользовательский файл: 46 378;
- суммарное улучшение: **82 хода**;
- изменено 39 строк, 1 004 строки исходного 46 378 сохранены без изменений;
- на момент работы публичный лидер имел 46 298, поэтому 46 296 удовлетворил установленному пользователем stop condition `score <= 46298`.

Ничего не было загружено и ничего не было отправлено на Kaggle. Вся сборка и проверка выполнены локально. После достижения 46 296 все оптимизаторы были остановлены.

## 1. Формализация задачи

Каждое состояние куба представлено вектором из 96 цветов. В `puzzle_info.json` находятся:

- `central_state` — solved colouring: по 16 значений каждого цвета `0..5`;
- 24 генератора — прямой и обратный quarter-turn для четырёх слоёв по трём осям `f`, `r`, `d`.

Ход применяется слева направо:

```python
state = tuple(state[i] for i in generator_permutation)
```

Путь — dot-separated строка, например `f0.-r2.d3`. Score равен сумме числа токенов во всех путях. Любая строка, которая не приводит ровно к `central_state`, делает submission некорректным. Wildcards нет.

Фактические файлы содержат 1 043 задачи, хотя текст overview в момент исследования ошибочно говорил о 1 042 задачах и 42 Santa-строках. Реально имеются IDs `0..1042`: первые 1 000 — generated walks, последние 43 — соответствующие 4×4×4 задачи Santa 2023.

Публичный leaderboard использует все test rows. Поэтому при неизменной метрике корректный локальный score равен leaderboard score; отправка submission для проверки не требовалась.

## 2. Главный принцип решения

Мы не пытались глобально перебрать группу 4×4×4. Вместо этого использовали портфельный подход:

1. Собрать как можно больше публично доступных корректных траекторий.
2. Реплеить каждую строку по официальным 96-позиционным перестановкам.
3. Для каждого ID сохранять самый короткий корректный путь.
4. Использовать разные траектории как множество промежуточных состояний.
5. Искать короткие exact-мосты между checkpoint-состояниями.
6. Применять bounded exact rewriting: coloured MITM, `twsearch --shortenseqs`, frame identities.
7. Принимать любое улучшение только после полного replay всей строки и всего итогового CSV.

Нейросети использовались только как эвристика ранжирования. Они ни разу не использовались как доказательство корректности.

## 3. Слой корректности

### 3.1 Валидатор

Основной валидатор — `src/cube444.py`. Он проверяет:

- все generator rows являются перестановками `0..95`;
- состояние имеет длину 96;
- IDs не повторяются;
- в submission нет missing/extra IDs;
- path не содержит неизвестных ходов или пустых токенов;
- ходы применяются строго слева направо;
- финальное состояние совпадает с `central_state` во всех 96 позициях;
- score считается по числу ходов, а не по длине строки.

Финальная команда:

```bash
python3 src/cube444.py outputs/cube4_submission_46296.csv \
  --puzzle-info data/puzzle_info.json \
  --test data/test.csv
```

Результат:

```text
puzzles=1043 score=46296 valid=True
```

### 3.2 Недоверенный candidate store

`src/merge_candidates.py` рассматривает любой внешний CSV/ZIP как недоверенный:

- ограничивает глубину и размер ZIP members;
- нормализует допустимые варианты названий колонок;
- независимо реплеит каждую уникальную пару `(ID,path)`;
- отбрасывает invalid rows;
- выбирает shortest solved path;
- при равной длине предпочитает baseline, чтобы не менять строки без причины;
- пишет output, provenance, rejections и JSON report атомарно.

Это было важно: широкий локальный harvest содержал 2 350 602 строки, из которых 798 731 оказались invalid. Ни одна из них не могла попасть в финал.

## 4. Публичные источники

### 4.1 Kaggle competition и notebooks

Использовались только read-only артефакты:

- [Competition](https://www.kaggle.com/competitions/cayley-py-444-cube/overview)
- [CayleyPy 2xT4 4x4x4 recommended](https://www.kaggle.com/code/trydotatwo/cayleypy-2xt4-4x4x4-recommended)
- [CayleyPy 2xT4 444 example](https://www.kaggle.com/code/trydotatwo/cayleypy-2xt4-444-example)

Код сбора:

- `src/collect_public_outputs.py` — небольшие статические outputs последних публичных запусков;
- `src/collect_kernel_history.py` — immutable notebook versions через versioned GET;
- `src/extract_kernel_history_candidates.py` — нормализация найденных путей;
- `src/kaggle_readonly.py` — только скачивание доступных submission, без upload/submit.

Последний public snapshot содержал 65 notebook refs. В первичном merge:

- 65 833 candidate rows были replay-valid;
- invalid rows: 0;
- 2 140 уникальных валидных путей;
- публичная база 46 718 улучшилась до 46 676.

Полный исторический harvest:

- 941 notebook version;
- 830 версий с candidate artifacts;
- 3 692 raw rows;
- 2 607 уникальных `(ID,path)`;
- ошибок сбора: 0.

Отдельно были проверены все версии двух главных notebooks:

- recommended: 277 версий;
- example: 96 версий;
- всего 373 версии, 289 с кандидатами;
- 5 500 raw rows, 1 304 уникальных пути;
- ни direct, ни trajectory-induced merge поверх 46 314 новых ходов уже не дали.

Отчёт: `work/reports/recommended_two_history_final_audit.json`.

### 4.2 GitHub result ledger

Источник: [TryDotAtwo/cayleypy-beam-results](https://github.com/TryDotAtwo/cayleypy-beam-results).

Это не solver repository. Это append-only ledger проверенных результатов и deterministic aggregate builder.

На snapshot, использованном до достижения stop condition:

- head: `bf3f68e5906e25ac7b8bfabef6c0d81bdfd1d309`;
- 490 canonical Cube444 JSON records;
- 459 уникальных `(ID,path)`;
- 385 покрытых ID;
- все 490 exact replay-valid;
- все использовали PieceTransformer/output-move-count и touch radius 4.

В прямом merge с публичной базой ledger дал 46 718 → 46 470. Ledger плюс последние notebook outputs дал 46 458.

Полный audit веток, PR refs и истории на этом snapshot не обнаружил скрытых или удалённых Cube444 records. На уровне 46 302: 305 IDs из ledger совпадали по длине, 80 были хуже, 0 лучше.

После остановки GitHub connector зафиксировал новый head `32a68babc4e4ad3e95ce22ee1bbb377f208e43a7`, на три commit-а впереди. Добавилась одна record для ID555 длины 44. Она не использовалась, потому что пользователь явно потребовал остановить работу при score ≤ 46 298.

### 4.3 Публичный CC0 submission corpus

Проверялся dataset `alexandervc/cayleypy-submissions`:

- 46 полных Cube444 submissions;
- лучший отдельный публичный score: 55 642;
- row-wise merge: 54 754;
- shallow exact rewriting давал только небольшой дополнительный выигрыш.

На поздних этапах этот corpus уже не содержал строк лучше текущего портфеля, но был полезен для разнообразия trajectories.

### 4.4 Santa 2023 trajectories

Два старых набора 4×4×4 путей были сопоставлены с текущими IDs не по fragile offset, а по exact reconstructed initial state:

```text
initial = solved · inverse(solution)
```

Получено 86 replay-valid trajectories для 43 IDs `1000..1042`; 14 legacy rows не сопоставились. Артефакты:

- `work/submissions/santa2023_mapped_444.csv`;
- `work/submissions/santa2023_itswin_mapped_444.csv`.

Они не дали strict gain в r5/r6 и targeted r7 bridge-search.

## 5. Полная хронология score

| Этап | Переход | Выигрыш | Изменённые ID | Механизм |
|---:|---:|---:|---|---|
| 1 | 46 378 → 46 332 | −46 | 66,153,503,504,509,510,525,623,629,633,634,635,643,673,675,794,797,799,815,816,818 | Direct merge публичных и локальных кандидатов |
| 2 | 46 332 → 46 330 | −2 | 495 | Direct merge параллельного локального портфеля |
| 3 | 46 330 → 46 328 | −2 | 505 | `twsearch` D8 |
| 4 | 46 328 → 46 322 | −6 | 550,586,704 | History direct + induced trajectory graph |
| 5 | 46 322 → 46 318 | −4 | 559,796 | `twsearch` D8 на historical alternatives |
| 6 | 46 318 → 46 316 | −2 | 575 | Exact coloured MITM r8 |
| 7 | 46 316 → 46 314 | −2 | 850 | `twsearch` D10 |
| 8 | 46 314 → 46 312 | −2 | 686 | Exact coloured MITM r8 |
| 9 | 46 312 → 46 302 | −10 | 213,497,549,630,659 | Direct merge sibling portfolio после frame rewrites |
| 10 | 46 302 → 46 296 | −6 | 97,133,346 | Direct merge пользовательского `cube444_merged_46372.csv` |

Контроль арифметики:

```text
46 + 2 + 2 + 6 + 4 + 2 + 2 + 2 + 10 + 6 = 82
46378 - 82 = 46296
```

Машинно-читаемая версия таблицы: `provenance/score_chronology.csv` в repro bundle.

## 6. Какие submission-файлы реально вошли в финал

Финал состоит из 1 004 неизменённых строк исходного `cube4_submission_46378.csv` и следующих 39 замен.

### 6.1 Первый merge: 46 378 → 46 332

`work/service_investigation/official_cube4_results.csv` дал 15 строк и −34:

| ID | Было | Стало | Δ |
|---:|---:|---:|---:|
| 503 | 48 | 46 | −2 |
| 509 | 48 | 46 | −2 |
| 510 | 47 | 45 | −2 |
| 623 | 46 | 44 | −2 |
| 629 | 48 | 46 | −2 |
| 633 | 48 | 46 | −2 |
| 635 | 46 | 44 | −2 |
| 643 | 46 | 44 | −2 |
| 673 | 48 | 46 | −2 |
| 675 | 48 | 46 | −2 |
| 794 | 47 | 45 | −2 |
| 797 | 50 | 44 | −6 |
| 799 | 48 | 46 | −2 |
| 815 | 46 | 44 | −2 |
| 818 | 47 | 45 | −2 |

`work/submissions/best_46452_twshort_d9.csv` дал 4 строки и −8:

- ID66: 47 → 45;
- ID153: 44 → 42;
- ID525: 44 → 42;
- ID816: 47 → 45.

`work/public_outputs/dingshijiana__minimal-cayleypy-notebook-for-cube4/cayleypy_public/output/solutions/all_solutions.csv`:

- ID634: 47 → 45.

`work/submissions/merged_public_twmerge.csv`:

- ID504: 47 → 45.

На этом этапе merger проверил 146 634/146 634 candidate rows; invalid rows: 0.

### 6.2 Остальные собственные и cross-workspace улучшения

| Источник | ID | Было → стало | Δ |
|---|---:|---:|---:|
| sibling `external/cayley6/outputs/submission_46330_valid.csv` | 495 | 42 → 40 | −2 |
| `work/candidates_tw_d8_high48.csv` | 505 | 48 → 46 | −2 |
| history direct | 550 | 47 → 45 | −2 |
| history direct | 586 | 47 → 45 | −2 |
| history induced-only | 704 | 47 → 45 | −2 |
| `work/candidates_tw_d8_history_partial.csv` | 559 | 48 → 46 | −2 |
| `work/candidates_tw_d8_history_partial.csv` | 796 | 47 → 45 | −2 |
| coloured r8 | 575 | 48 → 46 | −2 |
| `work/candidates_tw_d10_history_full.csv` | 850 | 47 → 45 | −2 |
| coloured r8 | 686 | 47 → 45 | −2 |

Только ID704 доказанно является новым путём, синтезированным induced graph, а не готовой строкой одного submission. IDs550 и 586 также присутствуют в induced result, но уже имели готовые более короткие historical paths.

### 6.3 Sibling portfolio: 46 312 → 46 302

Из локального параллельного `external/cayley6/outputs/submission_46304_valid.csv` вошли:

- ID213: 46 → 44;
- ID497: 46 → 44;
- ID549: 46 → 44;
- ID630: 47 → 45;
- ID659: 46 → 44.

Upstream evidence показывает, что эти строки были получены frame-local rewriting. Четыре появились в `external/cayley6/work/d10_alternatives_46314/slot_00_frame.csv`, затем ID213 был добавлен fixed-point frame pass.

### 6.4 Финальный пользовательский merge: 46 302 → 46 296

Из пользовательского файла `inputs/cube444_merged_46372.csv` взяты только три строки:

- ID97: 46 → 44;
- ID133: 40 → 38;
- ID346: 47 → 45.

Именно этот merge пересёк stop threshold. Upstream-история генерации этих трёх путей в локальных файлах отсутствует; доказано только, что исходный пользовательский CSV целиком replay-valid со score 46 372, а три выбранные строки короче текущего 46 302.

Полные final paths и их SHA-256 находятся в `provenance/final_replacements.csv`.

## 7. Собственные методы

### 7.1 Row-wise portfolio merge

Самый дешёвый и наиболее результативный метод: для каждого ID выбирать shortest replay-valid path из всех источников. Это не эвристическая оценка — каждая строка полностью проигрывается.

Код:

- `src/cube444.py`;
- `src/merge_candidates.py`;
- `src/build_candidate_slots.py`;
- `src/build_broad_tied_slots.py`.

Равнодлинные пути тоже сохранялись в отдельных slots, потому что разные промежуточные состояния полезны для последующего exact bridging.

### 7.2 Trajectory graph и induced edges

Код: `src/trajectory_graph_merge.py`.

Каждый checkpoint каждого корректного пути становится вершиной — exact 96-byte colour state. Реальные ходы добавляют legal edges. BFS от solved находит shortest path внутри конечного объединённого графа.

Режим `--induced-edges` проверяет для каждой уже существующей вершины все 24 хода. Если соседнее состояние тоже присутствует в собранном множестве, добавляется edge, даже если он не встречался в исходной траектории.

Гарантия: найденный путь кратчайший внутри этого конечного графа. Это не глобальный optimum куба.

Результаты:

- public compact direct: 46 718 → 46 458;
- induced на тех же states: 46 456, дополнительный −2 на ID525;
- history partial: direct дал IDs550/586, induced добавил ID704;
- более поздние induced graphs до 1 224 304 directed edges уже не улучшили 46 302.

### 7.3 Exact checkpoint splicing r4–r7

Код:

- `src/exact_state_splice.cpp`;
- `src/run_exact_slot_sweep.py`;
- `src/make_trajectory_bridge_walks.py`.

Для пути вычисляются checkpoint states `S0..Sn`. Между каждой парой checkpoints ищется более короткий bridge методом meet-in-the-middle:

- r4 = 2+2;
- r5 = 2+3;
- r6 = 3+3;
- r7 = 3+4.

Размеры exact balls:

- depth≤2: 493;
- depth≤3: 9 493;
- depth≤4: 182 407.

Сравниваются все 96 цветов. Hash используется только как индекс; collision всегда разрешается полным сравнением. Затем DAG-DP выбирает совместимый набор непересекающихся улучшений.

Положительный результат:

- ID153: 44 → 42, сначала exact r6, затем воспроизведён cross-r5.

Нулевые проверки:

- baseline 46 718: r4/r5/r6 — 0;
- broad diverse cross-r6: более 1.4 млрд candidate edges — 0 поверх текущего результата;
- D9 alternatives r6/r7 — 0;
- Santa r5/r6/targeted-r7 — 0;
- восемь tied slots exact r6 — 0.

### 7.4 Cross-trajectory algebra

Для primary path `P` и alternatives `Ai` строился корректный solving walk:

```text
P · A1^-1 · A1 · ... · Ak^-1 · Ak · P^-1 · P
```

Каждая inverse leg возвращает solved state к общему initial state. Поэтому checkpoints разных решений оказываются в одном корректном walk, после чего exact splicer может искать мосты между ними.

Forward и reverse layouts использовались для покрытия направлений. Greedy selection выбирал trajectories с максимальным числом новых checkpoint states.

### 7.5 Exact coloured MITM r8

Использованный solver: `external/helpers/colored_mitm8_changed.cpp`.

Оркестрация:

- `src/run_colored_slot0_batches.py`;
- `src/run_colored_slot1_batches.py`;
- `src/validate_colored_batches.py`.

Алгоритм строит exhaustive canonical words radius 4 с обеих сторон и проверяет 4+4 bridge между path checkpoints. Canonicalization использует только точные relations одного axis: commuting layers, степени modulo 4 и deterministic layer order.

Bloom filter — только no-false-negative prefilter. После hash hit всегда сравниваются все 96 цветов. Interval DP выбирает лучший набор непересекающихся окон.

Два production gains:

1. ID575, окно `[31,38)`, segment 7 → 5:

```text
-r1.f1.-f2.-d3.f2.r1.-f2
→
-f2.-r1.f1.-d3.r1
```

2. ID686, окно `[13,23)`, segment 10 → 8:

```text
f0.-r1.f2.f3.-d2.-f2.-d0.-f2.d2.f2
→
-d1.r2.d1.-r2.f0.-r1.f3.-d0
```

Большинство r8 scans дали 0. Например, odd slot0 проверил 312 distinct IDs за ~3 027 секунд без strict gain; even slot0 проверил 240 IDs за ~2 546 секунд и нашёл только ID686.

### 7.6 Coloured suffix/window MITM r10

Код:

- `src/colored_suffix_mitm10.cpp`;
- `src/make_mitm10_window_specs.py`;
- `src/rank_mitm10_windows.py`.

Строится 3 512 239 canonical words radius 5. Для incumbent window длины 11 или 12 exhaustively проверяется replacement длиной ≤9 или ≤10, то есть отсутствие gain≥2 на данном окне.

Результаты:

- все suffixes 11/12 на 46 316: 0, 824.99 с, peak RSS ~420 MiB;
- ML-ranked top50 внутренних окон: 0;
- ML-ranked top500: 0;
- все length-12 windows ID850: 0.

Это полезный отрицательный результат: shallow cleanup текущего портфеля почти полностью исчерпан.

### 7.7 `twsearch --shortenseqs`

Источник: [cubing/twsearch](https://github.com/cubing/twsearch), commit `f90bbc843a30a9fc22d7dd3ca3c441c5a77c7270`.

Код интеграции:

- `src/make_twsearch_cubie_definition.py`;
- `src/make_twsearch_definition.py`;
- `src/twsearch_optimize_submission.py`;
- `src/twsearch_optimize_candidates.py`.

Compact definition выводится из official group action, а не из вручную заданной геометрии. Она обнаруживает centers, paired wings и oriented corners. Каждый cubie generator декодируется обратно в 96 sticker destinations и обязан совпасть с official permutation.

Важно: `twsearch` переписывает тот же labelled cubie/group element. Это более строго, чем colour-state equivalence; поэтому coloured MITM способен найти некоторые сокращения, недоступные labelled rewriting.

Положительные результаты:

- D8: ID66, 47 → 45;
- D8 high48: ID505, 48 → 46;
- D8 history: IDs559 и 796, суммарно −4;
- D10 full history: ID850, 47 → 45.

D10 full history:

- 1 255 candidates;
- 4 008.5 с (~66.8 минуты);
- 19 alternatives стали короче;
- только ID850 улучшил incumbent;
- memory cap 8 GB, но фактический peak RSS не был записан.

### 7.8 24-frame symbolic DP

Идея: путь разбивается на maximal same-axis blocks. Из блока можно вынести общий exponent всех четырёх слоёв как whole-cube rotation. DP хранит одну из 24 pending orientations и переносит rotation через следующие блоки, корректно conjugating ходы. Конечная frame обязана быть identity.

Гарантия: DP оптимален внутри системы factor/carry/conjugate, а не во всей группе.

Результаты:

- slack12 corpus: 18 287 replay-valid paths, 491 source paths shortened;
- этот pass независимо нашёл ID686 и дал 46 304 → 46 302 в sibling workspace;
- slack20: 24 325 paths, 1 073 shortened source paths, но strict gain поверх 46 302 уже 0.

Код включён в bundle как `external/helpers/shorten_isometry_frames.py`; дополнительные sibling reports находятся в `external/cayley6`.

### 7.9 Colour-stabilizer identities

Исследовались слова `h`, которые фиксируют solved colour state, хотя являются nonidentity permutation labelled stickers. Тогда `P·h` остаётся colour-solution. Слова закрывались по inverse и cube automorphisms; single и pair joins проверялись exact replay.

Проверены D10 и D12 macro corpora, включая миллионы pair evaluations. Strict gain: 0. Метод полезен как отдельная перспективная ветка, но в score 46 296 ничего не внёс.

## 8. ML/beam методы

### 8.1 Public PieceTransformer

Архитектура:

- 56 piece tokens + CLS;
- `d_model=256`;
- 8 attention heads;
- 4 layers;
- FF=1024;
- ReLU;
- 24 Q outputs;
- 3 383 064 параметра.

Checkpoint SHA-256:

```text
58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164
```

Он совпал во всех 62 завершённых публичных run summaries. Эти публичные запуски использовали два Tesla T4, touch radius 4 и effective beam около 46–47 млн. Медиана solve time была около 37 450 секунд на одну задачу, поэтому локально повторять такой масштаб было неразумно.

### 8.2 Zenodo scalar agents

Источник: [Zenodo DOI 10.5281/zenodo.14886876](https://doi.org/10.5281/zenodo.14886876), официальный код [khoruzhii/cayleypy-cube](https://github.com/khoruzhii/cayleypy-cube), commit `f02604fa7b665b82e5fbe5692b336a4fe4a01bdc`.

Скачаны agents 01, 11 и 24 через strict HTTP Range GET. Для agent24:

- compressed: 14 991 868 B;
- uncompressed: 16 089 394 B;
- CRC32: `af81fd94`;
- SHA-256: `054cc6a39f310178cb7904c29a5800b84b66628cb60fa35361be55f6342a7287`.

Локальный `src/scalar_beam444.py` поддерживает:

- MPS/CPU;
- inverse pruning;
- exact dedup;
- touch-BFS;
- suffix anchors;
- rotation views;
- multi-agent quota ensemble;
- обязательный exact replay.

56 bounded attempts:

- 717.15 с;
- 61 692 464 expanded states;
- 59 289 027 scored states;
- ни одного решения короче incumbent.

То есть scalar agents не дали прямого вклада в финальный score, но использовались для ранжирования окон и оценки перспективных branches.

### 8.3 Q-model bundle

Проверялся публичный bundle `artgor/cube444-q-beam-tpu-assets`:

- `s3.npz`: 12 611 753 B, PieceTransformer Q;
- `mlp_x16.npz`: 174 543 009 B, PairQMLP;
- blend: `0.6*s3 + 0.4*mlp_x16`;
- bundled submission: 46 662.

Критичные contract notes:

- `num_classes=6`, не 96;
- это colour cube, поэтому нельзя слепо применять `invert_state`/NISS;
- symmetry требует slot rotation вместе с recolouring;
- NPZ weights имеют JAX-oriented `(in,out)` layout;
- MLP BatchNorm folded и корректен только в eval.

`src/qbeam444_npz.py` реализует loader и exact-bounded beam infrastructure, но полноценный новый search run не был завершён: во время подготовки пришёл пользовательский файл, direct merge сразу дал 46 296, после чего stop condition потребовал прекратить вычисления. Поэтому нельзя утверждать, что этот Q-port дал решение.

## 9. Что не сработало и почему это важно

Нулевые результаты — часть решения, потому что они исключили дешёвые направления:

- same-axis normalization и shallow group rewrites почти ничего не давали;
- full-state exact splice r4/r5/r6 на сильной базе — 0;
- broad r6/r7 cross-trajectory passes — почти всегда 0;
- r10 suffix/window MITM — 0;
- scalar-agent beams — 0;
- public Q submission 46 662 не улучшил 46 302;
- all-history direct и induced merges после 46 318 — 0;
- Santa trajectories r5/r6/r7 — 0;
- stabilizer single/pair macros — 0;
- GitHub ledger refresh поверх 46 302 — 0;
- полный аудит 373 версий двух главных notebooks поверх 46 314 — 0.

Практический вывод: сильный portfolio уже локально неприводим на малых радиусах. Реальные поздние выигрыши приходили либо из очень разнообразных внешних trajectories, либо из более дорогих D8/D10/coloured-r8 переписываний.

## 10. Файлы кода

### Core validation и merge

- `src/cube444.py` — exact validator и scorer;
- `src/merge_candidates.py` — безопасный portfolio merger;
- `src/export_writeup_provenance.py` — транзитивный provenance финала;
- `tests/test_merge_candidates.py` — тесты merger.

### Public artifact collection

- `src/collect_public_outputs.py`;
- `src/collect_kernel_history.py`;
- `src/extract_kernel_history_candidates.py`;
- `src/kaggle_readonly.py`.

### Portfolio/trajectory

- `src/build_candidate_slots.py`;
- `src/build_broad_tied_slots.py`;
- `src/trajectory_graph_merge.py`;
- `src/make_trajectory_bridge_walks.py`;
- `src/map_santa444_paths.py`;
- `src/make_submission_overrides.py`;
- `src/filter_shortened_candidates.py`.

### Exact MITM

- `src/exact_state_splice.cpp`;
- `src/run_exact_slot_sweep.py`;
- `external/helpers/colored_mitm8_changed.cpp`;
- `src/run_colored_slot0_batches.py`;
- `src/run_colored_slot1_batches.py`;
- `src/validate_colored_batches.py`;
- `src/colored_suffix_mitm10.cpp`;
- `src/make_mitm10_window_specs.py`;
- `src/rank_mitm10_windows.py`.

### twsearch

- `src/make_twsearch_cubie_definition.py`;
- `src/make_twsearch_definition.py`;
- `src/twsearch_optimize_submission.py`;
- `src/twsearch_optimize_candidates.py`;
- `external/twsearch/` — source snapshot без `.git` и build artifacts.

### Neural

- `src/fetch_scalar_agent.py`;
- `src/scalar_beam444.py`;
- `src/qbeam444_npz.py`;
- `external/models/pilgrim/`;
- `external/models/unified_training/`.

## 11. Репродукция финального blend

Этот репозиторий является распакованным reproducibility bundle и содержит:

- исходные пользовательские 46 378 и 46 372;
- все 10 milestone submissions;
- candidate CSV, реально давшие 39 замен;
- JSON/CSV provenance и relation tables;
- final CSV;
- core source code;
- exact C++ sources;
- twsearch source snapshot;
- ключевые audit reports;
- rebuild script.

Из корня bundle:

```bash
./REBUILD_FINAL.sh
```

Скрипт последовательно воспроизводит score:

```text
46378 → 46332 → 46330 → 46328 → 46322
      → 46318 → 46316 → 46314 → 46312
      → 46302 → 46296
```

Затем он:

1. exact-replay проверяет 1 043 строки;
2. проверяет score 46 296;
3. требует byte-identical совпадение с архивным final CSV.

Тест rebuild прошёл успешно.

Для повторного экспорта provenance:

```bash
python3 src/export_writeup_provenance.py
```

## 12. Ограничения утверждений

Score 46 296 полностью проверен. Но это не доказательство глобальной оптимальности каждого из 1 043 путей.

Точные формулировки:

- final validity — exact;
- rowwise shortest среди собранного portfolio — exact;
- BFS shortest внутри собранного trajectory graph — exact;
- MITM completeness — exact только в указанном радиусе/окне;
- frame DP — exact только внутри своей algebraic rewrite system;
- neural ranking/beam — heuristic;
- leaderboard superiority зависит от актуального состояния leaderboard в момент submission.

## 13. Безопасность и политика сети

- Ни один наш процесс не вызывал Kaggle submit.
- Ни один наш процесс не загружал notebook output.
- Publisher в публичном notebook не запускался.
- Для GitHub ledger использовались read-only GET/connector operations.
- Внешний POST publisher был намеренно исключён.
- Пользовательский Kaggle token не включён ни в код, ни в отчёт, ни в bundle.

Поскольку token был вставлен непосредственно в чат, его разумно отозвать и выпустить новый.

## 14. Финальные артефакты

- `outputs/cube4_submission_46296.csv` — готовый submission;
- `README_RU.md` — этот полный отчёт;
- `README.md` — краткая англоязычная страница репозитория;
- `provenance/score_chronology.csv` — score history;
- `provenance/final_replacements.csv` — все 39 замен с полными путями;
- `provenance/provenance_summary.json` — итоговая машинная сводка;
- остальная структура репозитория — распакованный reproducibility bundle.
