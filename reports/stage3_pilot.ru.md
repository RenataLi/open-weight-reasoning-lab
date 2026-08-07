# Stage 3: зафиксированный multi-seed SFT confirmation

## Краткий итог

Stage 3 проверял, может ли reasoning-aware SFT mixture устранить большую потерю capability из answer-only Stage 2 и одновременно выучить проверяемый формат ответа. Поиск конфигурации выполнялся на трёх seed только по validation-данным; до открытия confirmation был записан SHA-256 selection lock; затем выбранная конфигурация один раз оценена на новых данных.

Выбранный QLoRA-рецепт сохранил точность Base внутри заранее заданных non-inferiority границ на всех seed. На 512 новых synthetic задачах Base получил 99,4%, SFT — в среднем 98,4% ± 0,3%; валидный final-answer format вырос с 22,7% до 100%. На 1 191 ранее не использованной задаче GSM8K Base получил 65,5%, каждый SFT seed — 65,2%. Прошли все проверки confidence bounds, parsing, token limits, continued-SFT и seed stability.

Статус: **МОЖНО ПРОЕКТИРОВАТЬ ОГРАНИЧЕННЫЙ DPO PILOT**. Это подтверждение стабильной SFT-основы, а не доказательство улучшения reasoning или пользы DPO.

## Исследовательский вопрос и зафиксированный gate

Основной вопрос — non-inferiority: можно ли научить модель краткому reasoning с финальной строкой ответа без большой потери format-independent capability, обнаруженной в Stage 2?

До просмотра confirmation были зафиксированы границы:

- regression synthetic exact match не хуже `-5 п.п.` для каждого seed;
- regression GSM8K exact match не хуже `-3 п.п.`;
- нижние границы paired-bootstrap 95% CI выше тех же `-5 / -3 п.п.`;
- synthetic valid final-answer format не ниже `95%`;
- parse success не ниже `99%` на обоих наборах;
- generation-limit rate не выше `5%` synthetic и `10%` external;
- continued-SFT regression относительно SFT-parent не хуже `-5 п.п.`;
- диапазон между seed не выше `3 п.п.` synthetic и `4 п.п.` external.

Доля GSM8K-ответов с финальным `#### number` показана как описательная метрика, а не как gate: правильность считается отдельно, с benchmark-specific fallback по последнему числу; parse success при этом обязателен.

## Диагностика Stage 2

Answer-only SFT в Stage 2 изменил synthetic exact match с `95,3%` до `71,9%`, strict format — с `0%` до `100%`, а среднюю длину ответа — примерно со `108` до `6` токенов. Сильнее всего пострадали многошаговые семейства. Это поддерживает, но не доказывает гипотезу, что модель 1.7B использовала видимые вычисления как часть решения, а answer-only supervision их подавил.

Stage 3 поэтому варьировал долю reasoning targets и prompt/response contract. Внутри каждого search сохранялись модель, LoRA shape, число примеров, step budget и seeds. Reasoning completions длиннее, поэтому эксперимент matched по примерам и шагам, но не по числу supervised tokens или compute.

## Целостность данных

| Роль | Количество | Источник | SHA-256 |
|---|---:|---|---|
| Synthetic validation | 512 | детерминированное расширение `arithmetic_v2` | сохранён в search run |
| GSM8K development | 256 | фиксированный split из train | сохранён в search run |
| Synthetic confirmation | 512 | новые детерминированные ID | `1913068be66d4d27991e79ec2ab89e39e0e613e653bf84c3336659d606c11f0f` |
| GSM8K confirmation | 1 191 | official test без 128 примеров Stage 2 | `a575174cb5d5f9a8adec03319b869ee902a6d44ba77ebc070a212d72ff7a29c9` |

Row-ID set hashes: `123b91822bcbb0be3fcaab10bb7d0eebed1c0954b0f758fd6627053c471f0b53` для synthetic и `12a858787fa44a98c7cb6186cd24801c6333a386a576dafe81e843441503cded` для GSM8K. Независимая проверка показала нулевое пересечение selection и confirmation, а также нулевое пересечение Stage 2 held-out и Stage 3 confirmation.

GSM8K зафиксирован на revision `740312add88f781978c0658806c59bc2815b9866`. Он не использовался для SFT; во время выбора был виден только фиксированный dev split из train.

## Поиск и отрицательные результаты

| Search | Output contract | Итог | Основной diagnostic result |
|---|---|---|---|
| v1 | Shared prompt, answer-only evaluation | Нет прошедших | 75% reasoning сохранили exactness, но mean strict correct лишь 14,5% |
| v2 | Contract-conditioned, answer-only evaluation | Нет прошедших | 90% reasoning: worst retention -15,0 п.п., strict correct 38,6% |
| v3 | Reasoning, затем final answer | Два прошедших | Contract-conditioned mixtures устранили validation trade-off |

У v1/v2 Base synthetic был 93,9%, у v3 — 98,4%, потому что различаются prompts и output contract. Эти значения нельзя напрямую сравнивать как одну метрику.

Финальная v3 matrix:

| Candidate | Прошёл на всех seed | Worst-seed retention | Mean synthetic strict correct | Mean GSM8K-dev exact |
|---|---:|---:|---:|---:|
| `shared75_nll_lr5e5` | нет | -1,8 п.п. | 91,4% | 76,6% |
| `conditioned75_nll_lr5e5` | да | -1,8 п.п. | 96,7% | 77,7% |
| `conditioned90_nll_lr5e5` | да | -1,6 п.п. | 96,9% | 77,0% |
| `shared75_nll_lr2e5` | нет | -8,2 п.п. | 79,8% | 76,6% |

Заранее заданный ranking сначала максимизировал worst-seed retention, поэтому выбрал `conditioned90_nll_lr5e5`, а не максимальный mean GSM8K-dev. Финальная v3 matrix содержит только NLL; сравнение objectives по ней не заявляется. DFT присутствовал лишь в более раннем exploratory protocol с другим контрактом и не дал eligible candidate.

## Selection lock

- Candidate: `conditioned90_nll_lr5e5`.
- Mixture: 90% verified concise reasoning, 10% direct response.
- Prompt conditioning: `contract_conditioned`.
- Loss / learning rate / steps: NLL / `5e-5` / `32`.
- QLoRA: NF4 double quantization, BF16 compute, rank 16, alpha 32, dropout 0.05, all-linear targets, gradient checkpointing.
- Seeds: `101`, `202`, `303`.
- Base revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Config hash: `ebfdb4683c864517fe7f9ad0e214d7fcdafb195b019213d850c48b786ec86689`.
- Selection lock hash: `20b23bc3b08bd19ec948da6384f4309179f2374423a3e7cdb96342d4a9df97bf`.

Lock записан `2026-08-06T18:36:43.874946Z`, test marker — `2026-08-06T18:37:43.209633Z`, примерно на 59 секунд позже. В lock явно записано `test_metrics_used_for_selection: false`.

## Held-out confirmation

| Model | Seed | Synthetic exact | Synthetic format | GSM8K exact | GSM8K `####` format | GSM8K strict correct | GSM8K token-limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | — | 99,4% | 22,7% | 65,5% | 88,9% | 60,4% | 3,7% |
| SFT | 101 | 98,6% | 100,0% | 65,2% | 89,6% | 59,9% | 2,8% |
| SFT | 202 | 98,0% | 100,0% | 65,2% | 88,6% | 59,4% | 2,9% |
| SFT | 303 | 98,4% | 100,0% | 65,2% | 87,6% | 58,6% | 2,4% |
| Continued-SFT | 101 | 98,4% | 100,0% | 65,2% | 92,2% | 61,8% | 2,9% |
| Continued-SFT | 202 | 98,4% | 100,0% | 66,3% | 90,5% | 61,9% | 2,2% |
| Continued-SFT | 303 | 98,6% | 100,0% | 65,8% | 91,6% | 62,0% | 2,5% |

Во всех ячейках parse success равен 100%; synthetic generation ни разу не достиг generation limit.

### Paired Base → SFT

| Seed | Synthetic delta | Synthetic 95% CI | GSM8K delta | GSM8K 95% CI | GSM wrong→right / right→wrong |
|---:|---:|---:|---:|---:|---:|
| 101 | -0,78 п.п. | [-1,76; 0,00] | -0,34 п.п. | [-2,43; +1,68] | 75 / 79 |
| 202 | -1,37 п.п. | [-2,54; -0,20] | -0,34 п.п. | [-2,43; +1,76] | 77 / 81 |
| 303 | -0,98 п.п. | [-2,15; 0,00] | -0,34 п.п. | [-2,43; +1,76] | 78 / 82 |

Synthetic результат non-inferior, но не лучше Base. Для seed 202 интервал не включает ноль и показывает небольшую потерю на этой выборке, однако нижняя граница намного выше locked guardrail `-5 п.п.`. На GSM8K интервалы исключают regression хуже `-3 п.п.`, но не доказывают equivalence или improvement.

Одинаковый aggregate GSM8K exact скрывает churn: каждый adapter меняет результат примерно на 154–160 задачах. Поэтому paired transitions обязательны в следующих этапах.

### Слабые strata

На difficulty 1 все SFT seed сохранили 100%, на difficulty 2 — 99,4%. Difficulty 3 снизился до `96,6%`, `95,0%`, `96,1%` при Base `98,3%`. Multiplication — худшее семейство с delta от `-1,6` до `-3,3 п.п.`. Эти strata должны стать preregistered abort checks DPO pilot.

## Continued-SFT control

Каждый SFT adapter получил независимую 16-step ветку на одном locked continuation mixture из 256 примеров. Записаны parent-adapter hash, Base revision, новый optimizer и scheduler, initial global step 0, отсутствие checkpoint resume и continuation hash `dcebb4880d3399ec4688f7af9dc26d32b9383cdebc007674dda7be109a29cb4f`.

Synthetic deltas SFT → continued: `-0,20`, `+0,39`, `+0,20 п.п.`. GSM8K: `0,00`, `+1,18`, `+0,67 п.п.`. Это stress test стабильности, а не доказательство пользы продолжения обучения.

## Gate audit

| Проверка | Наблюдение | Порог | Итог |
|---|---|---|---:|
| Lock раньше test | около 59 секунд | строго раньше | PASS |
| Locked seeds | 3 / 3 | 3 / 3 | PASS |
| Synthetic regression | от -0,78 до -1,37 п.п. | не хуже -5 п.п. | PASS |
| GSM8K regression | -0,34 п.п. | не хуже -3 п.п. | PASS |
| Worst synthetic CI lower | -2,54 п.п. | выше -5 п.п. | PASS |
| Worst GSM8K CI lower | -2,43 п.п. | выше -3 п.п. | PASS |
| Synthetic format | 100% | не ниже 95% | PASS |
| Parse success | 100% | не ниже 99% | PASS |
| Token-limit | 0% / 2,4–2,9% | не выше 5% / 10% | PASS |
| Seed range | 0,59 / 0 п.п. | не выше 3 / 4 п.п. | PASS |
| Continued retention | worst -0,20 п.п. | не хуже -5 п.п. | PASS |

Итог: **МОЖНО ПРОЕКТИРОВАТЬ ОГРАНИЧЕННЫЙ DPO PILOT**.

## Независимый integrity audit

После run были заново пересчитаны counts из raw predictions, проверены role tags, data/adapter hashes, нулевые split overlaps и lineage всех continued-SFT веток. Числа отчёта полностью совпали с сырыми строками.

Аудит также обнаружил слабости первоначального runner: повторный confirm не блокировался, selection lock и artifact maps покрывали не все evidence files, а gate доверял части provenance fields вместо повторной проверки. Код для будущих запусков усилен без изменения порогов и без пересчёта model outputs. Historical run остаётся неизменённым; отдельно указано, что было locked изначально и что проверено post hoc.

Confirmation sets теперь считаются раскрытыми. Их нельзя использовать для настройки DPO или снова называть fresh.

## Следующий корректный эксперимент

Stage 3 успешно сохранил capability и научил контракту, но reasoning gain не показан. Более того, Base уже решает 509 из 512 fresh synthetic tasks, поэтому набор почти упёрся в потолок.

DPO feasibility pilot должен включать:

1. один SFT seed, выбранный заранее, а не лучший по confirmation;
2. preference pairs только из train-only или новых prompts;
3. matched extra-SFT baseline на chosen responses;
4. length-bias, KL, response-length и token-cap diagnostics;
5. более сложный fresh synthetic dev и отдельный untouched final holdout;
6. beta, learning rate, seed и decoding, зафиксированные до final evaluation;
7. abort strata: multiplication, difficulty 3, GSM8K format/strict и example churn.

## Воспроизведение

Проверенная среда: Windows 11, Python 3.12.13, NVIDIA RTX 5090, PyTorch 2.13.0+cu130, Transformers 5.14.1, TRL 1.9.2, PEFT 0.20.0, bitsandbytes 0.50.0.

Search run: `20260806_211000_qwen3-1.7b-stage3-reasoning-contract-search`.

Confirmation run: `20260806_213800_qwen3-1.7b-stage3-reasoning-contract-confirm`.

```powershell
.\.venv\Scripts\python.exe -m reasoning_lab.cli run-stage3-search `
  --config configs\stage3_reasoning_contract_qwen3_1_7b.yaml

.\.venv\Scripts\python.exe -m reasoning_lab.cli run-stage3-confirm `
  --config configs\stage3_reasoning_contract_qwen3_1_7b.yaml `
  --search-run runs\<completed-search-run>
```

Компактные publication-safe значения находятся в [`stage3_summary.json`](stage3_summary.json). Raw predictions, metrics, manifests, adapters и success/test markers остаются в локальном ignored `runs/`.

