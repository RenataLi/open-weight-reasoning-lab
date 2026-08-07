<div align="center">
  <img src="docs/assets/hero.svg" width="100%" alt="Reasoning Post-Training Lab — исследовательский pipeline QLoRA SFT и DPO">
</div>

<p align="center">
  <a href="README.md">English version</a>
</p>

<p align="center">
  <a href="https://github.com/RenataLi/open-weight-reasoning-lab/actions/workflows/core-tests.yml"><img alt="Core tests" src="https://github.com/RenataLi/open-weight-reasoning-lab/actions/workflows/core-tests.yml/badge.svg"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Transformers" src="https://img.shields.io/badge/Transformers-5.14-FFD21E?logo=huggingface&logoColor=111827">
  <img alt="TRL + PEFT" src="https://img.shields.io/badge/TRL%20%2B%20PEFT-DPO%20%7C%20QLoRA-7C3AED">
  <img alt="Тесты" src="https://img.shields.io/badge/tests-87%20passed-22C55E">
  <img alt="Лицензия" src="https://img.shields.io/badge/license-Apache--2.0-0EA5E9">
</p>

# Reasoning Post-Training Lab

Воспроизводимый локальный исследовательский стек для **QLoRA supervised fine-tuning** и **Direct Preference Optimization** модели [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) на математических, логических и структурированных reasoning-задачах.

Post-training здесь реализован как контролируемый эксперимент: детерминированные данные, точные ревизии модели и датасетов, выбор только по validation, неизменяемые SHA-256 locks, matched extra-SFT контроли, paired-оценка и машиночитаемые доказательства для каждого запуска.

## Главное в цифрах

| Исследовательский результат | Значение |
|---|---:|
| Свежая held-out оценка Stage 3 | **1 703 задачи** |
| Зафиксированные SFT seeds | **3** |
| Соблюдение synthetic final-answer формата | **100,0%** |
| Synthetic exact match после SFT | **98,4% ± 0,3%** |
| Пройденные DPO development gates | **15 / 15** |
| Preference accuracy выбранного DPO | **64,1%** |
| Автоматические тесты | **87 passed** |

## Проверенные результаты

### Stage 3 — multi-seed held-out подтверждение SFT

Зафиксированный рецепт `conditioned90_nll_lr5e5` использует contract-conditioned prompts и 90% проверенных concise-reasoning targets. Он был выбран на development-данных, закреплён неизменяемым lock и затем оценён на seeds `101`, `202`, `303` по **512 synthetic + 1 191 GSM8K задачам**.

| Метрика на held-out данных | Base | QLoRA SFT, среднее ± SD | Fresh continued-SFT, среднее ± SD |
|---|---:|---:|---:|
| Synthetic exact match | 99,4% | **98,4% ± 0,3%** | **98,5% ± 0,1%** |
| Валидный synthetic final-answer формат | 22,7% | **100,0% ± 0,0%** | **100,0% ± 0,0%** |
| GSM8K exact match | 65,5% | **65,2% ± 0,0%** | **65,8% ± 0,6%** |
| GSM8K parse success | 100,0% | **100,0% ± 0,0%** | **100,0% ± 0,0%** |

<p align="center">
  <img src="docs/assets/stage3-results.svg" width="920" alt="Результаты held-out оценки Stage 3">
</p>

Все заранее заданные capability, format, uncertainty и seed-stability gates пройдены. Независимый integrity-аудит подтвердил порядок selection/confirmation, нулевое пересечение splits, lineage адаптеров и хеши артефактов.

### Stage 4 — safety-first DPO development search

Stage 4 сравнивает три заранее зафиксированных DPO-варианта с двумя fresh-state extra-SFT контролями. Все три DPO-ветки прошли полный набор development gates. Правило выбора остановилось на low-learning-rate варианте `dpo_sigmoid_lr2e6` благодаря лучшему worst-task retention.

| Зафиксированная development-метрика | Stage 3 parent | Выбранный DPO | Выбранный extra-SFT control |
|---|---:|---:|---:|
| Synthetic exact match | 69,8% | **70,3%** | 72,9% |
| Synthetic strict correct | 64,6% | **64,6%** | 71,4% |
| SVAMP exact match | 75,4% | **76,4%** | 77,4% |
| Chosen-over-rejected accuracy | — | **64,1%** | 38,3% |
| Parse success | 100,0% | **100,0%** | 100,0% |

Выбранная DPO-policy повысила exact match относительно замороженного parent на обоих reasoning-наборах и превзошла выбранный extra-SFT comparator по разделению chosen/rejected на **25,8 процентного пункта**. Соотношение длины генераций к parent осталось близким к единице: `0,996×` на synthetic и `1,025×` на SVAMP.

<p align="center">
  <img src="docs/assets/stage4-development.svg" width="920" alt="Development scorecard Stage 4 DPO">
</p>

Подробности: [очищенный отчёт Stage 4](reports/stage4_development_search.md) и [машиночитаемое aggregate summary](reports/stage4_development_summary.json).

## Чем проект выделяется

- **Experiment integrity реализован в коде.** Search и confirmation разделены на разные команды; selection locks неизменяемы, а открытие confirmation — одноразовое.
- **Криптографический provenance.** Сохраняются resolved configs, точные revisions, хеши файлов и множеств строк, source attestations, lineage адаптеров и snapshots окружения.
- **Matched controls.** DPO сравнивается с update-matched и token-proxy continued-SFT ветками из одного parent.
- **Capability-aware selection.** Preference accuracy оценивается вместе с exact-match retention, форматом, strict correctness, длиной и worst-family regression.
- **Структурированная диагностика.** Метрики разбиваются по семействам и сложности; там, где применимо, рассчитываются paired transitions и bootstrap intervals.
- **Локальная эффективность.** NF4 QLoRA, bf16, gradient checkpointing, LoRA-адаптеры и precomputed reference log-probabilities позволяют выполнять pipeline на одной GPU.

## Исследовательский workflow

<p align="center">
  <img src="docs/assets/pipeline.svg" width="100%" alt="Полный жизненный цикл эксперимента">
</p>

Synthetic benchmark охватывает семь семейств: arithmetic chains, exact division, linear equations, logic/code, percentage change, ratio allocation и table aggregation. Они распределены по математическому, логическому и структурированному доменам и трём уровням сложности. Preference pairs используют детерминированные локальные искажения шагов, баланс higher/lower ошибок и семантическую дедупликацию.

## Архитектура

```text
src/reasoning_lab/
├── data/                    генераторы и versioned dataset snapshots
├── training/                QLoRA SFT, continued-SFT и DPO
├── evaluation/              generation, parsing, metrics, preference scoring
├── reporting/               Markdown и machine-readable отчёты
├── experiment_integrity.py  hashes, attestations, locks, exposure registry
├── stage*_selection.py      зафиксированные gates и ranking rules
├── stage*_pipeline.py       end-to-end orchestration
└── cli.py                   командный интерфейс
```

## Быстрый запуск

### 1. Создать окружение

```powershell
git clone https://github.com/RenataLi/open-weight-reasoning-lab.git
cd open-weight-reasoning-lab

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

### 2. Проверить GPU stack

```powershell
reasoning-lab preflight
```

Preflight проверяет CUDA, bf16, bitsandbytes NF4, загрузку tokenizer и небольшой forward pass до начала обучения.

### 3. Запустить компактный end-to-end pipeline

```powershell
reasoning-lab generate-data --config configs/e2e_smoke.yaml
reasoning-lab run --config configs/e2e_smoke.yaml
```

### 4. Запустить исследовательские stages

```powershell
# Controlled QLoRA pilot
reasoning-lab run-stage2 --config configs/stage2_qwen3_1_7b.yaml

# Validation-only multi-seed SFT search
reasoning-lab run-stage3-search `
  --config configs/stage3_reasoning_contract_qwen3_1_7b.yaml

# Development-only DPO + matched-control search
reasoning-lab run-stage4-search `
  --config configs/stage4_dpo_qwen3_1_7b.yaml
```

Поздние конфиги намеренно привязаны к точным upstream run IDs и хешам. Для новой lineage сначала воспроизводится предыдущий stage, затем его anchors фиксируются в новой версии протокола.

## Контракт воспроизводимости

Каждый полный запуск создаёт audit-ready bundle:

- resolved configuration и config SHA-256;
- точные model/dataset revisions и seeds;
- environment и GPU preflight snapshots;
- trainable-parameter manifests и adapter hashes;
- data-file, row-ID-set, prompt-set и semantic-set hashes;
- метрики по семействам задач и сложности;
- selection records, immutable lock anchors и status markers;
- source-code attestation и manifest ключевых артефактов.

Тяжёлые локальные артефакты намеренно исключены из Git: model caches, generated/external datasets, raw runs, adapters, checkpoints, predictions, logs и закрытые confirmation rows. В репозитории находятся код, конфиги, тесты, очищенные aggregate-отчёты и готовые визуализации.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Полный локальный suite содержит **87 проходящих тестов**: config validation, deterministic generation, split integrity, parsing, metrics, checkpoints, SFT/DPO lineage, selection gates, hash verification и one-time confirmation semantics. GitHub Actions запускает лёгкое CPU-only ядро на каждый push и pull request.

## Отчёты

- [Stage 1 pilot](reports/stage1_pilot.md)
- [Stage 2 controlled QLoRA pilot](reports/stage2_pilot.md)
- [Stage 3 multi-seed confirmation](reports/stage3_pilot.ru.md)
- [Stage 3 machine-readable summary](reports/stage3_summary.json)
- [Stage 3 integrity audit](reports/stage3_integrity_audit.json)
- [Stage 4 DPO development search](reports/stage4_development_search.md)
- [Stage 4 aggregate summary](reports/stage4_development_summary.json)

## Основные источники

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [GSM8K](https://arxiv.org/abs/2110.14168)
- [SVAMP](https://aclanthology.org/2021.naacl-main.168/)

## Лицензия

Apache License 2.0 — см. [LICENSE](LICENSE).

Машиночитаемые данные для цитирования: [CITATION.cff](CITATION.cff).
