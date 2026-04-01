# AERO / Route1 Open-Set Domain Generalization Framework

This repository contains the implementation and experiment scripts used for the TII manuscript on **AERO (Route1)** under open-set domain generalization for bearing fault diagnosis.

Current public scope in this folder:
- Main scenarios: **Setting B** (cross-domain) and **Setting C** (cross-dataset)
- Main datasets: **CWRU** and **JNU**
- Newly added baselines: **AFRDN** and **WPA-PDG**
- New control ablation: **EQCAP_NO_ENV** (equal-capacity control)
- OOD score comparison: **MSP / Energy / MaxLogit**
- Pseudo-environment interpretability analysis and visualization

## 1. Repository Layout

```text
route1_framework/
├── algorithms/                     # Route1 + baseline training logic
├── datasets/                       # CWRU/JNU + optional SELF loaders
├── models/                         # Route1 and baseline model definitions
├── scripts/                        # aggregation / plotting / analysis scripts
├── run_scenario_benchmark.py       # scenario runner (A/B/C)
├── run_ablation_benchmark.py       # ablation runner (including EQCAP_NO_ENV)
├── train.py / test.py              # Route1 train/test
├── train_baseline.py / test_baseline.py
├── BASELINE_AFRDN_WPAPDG_README_CN.md
├── ABLATION_EQCAP_README_CN.md
├── OOD_README_CN.md
└── PSEUDOENV_ALIGNMENT_README_CN.md
```

## 2. Environment

Recommended:
- Python `3.9` or `3.10`
- PyTorch with CUDA matching your server

Install dependencies:

```bash
cd 代码/route1_framework
pip install -r requirements.txt
```

If your environment has `GLIBCXX` issues, prepend:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## 3. Data Preparation

Default dataset roots are configured in `config.py`:
- `数据集/CWRU/参考文献`
- `数据集/JNU/JNU-Bearing-Dataset-main`

For SELF datasets, pass runtime args instead of hard-coding:
- `--self_processed_root /path/to/Processed_Data_Numpy`
- expected subfolders: `SelfA/`, `SelfB/`

## 4. Quick Reproduction (Paper-Core)

### 4.1 New baselines only (AFRDN + WPA-PDG), CWRU->JNU direction

```bash
bash scripts/run_ab_c_multiseed.sh \
  --new_methods_only \
  --repeat_strategy both \
  --unknown_list 1,2,3 \
  --seeds 41,42,43 \
  --direction c2j \
  --exp_prefix tii319_afrdn_wpa_c2j
```

### 4.2 New baselines only (AFRDN + WPA-PDG), JNU->CWRU direction

```bash
bash scripts/run_ab_c_multiseed.sh \
  --new_methods_only \
  --repeat_strategy both \
  --unknown_list 1,2,3 \
  --seeds 41,42,43 \
  --direction j2c \
  --exp_prefix tii319_afrdn_wpa_j2c
```

### 4.3 Equal-capacity control ablation (FULL vs EQCAP_NO_ENV)

CWRU direction:

```bash
python run_ablation_benchmark.py \
  --scenario B_AND_C \
  --dataset CWRU --domains 0HP,1HP,2HP,3HP \
  --train_dataset CWRU --train_domains 0HP,1HP,2HP,3HP \
  --test_dataset JNU --test_domains 600,800,1000 \
  --known_classes 0,1,3 --unknown_classes 2 \
  --epochs 20 --batch_size 64 --m_max 15 \
  --seeds 41,42,43 \
  --variants FULL,EQCAP_NO_ENV \
  --exp_name ablation_eqcap_cwru
```

JNU direction:

```bash
python run_ablation_benchmark.py \
  --scenario B_AND_C \
  --dataset JNU --domains 600,800,1000 \
  --train_dataset JNU --train_domains 600,800,1000 \
  --test_dataset CWRU --test_domains 0HP,1HP,2HP,3HP \
  --known_classes 0,1,3 --unknown_classes 2 \
  --epochs 20 --batch_size 64 --m_max 15 \
  --seeds 41,42,43 \
  --variants FULL,EQCAP_NO_ENV \
  --exp_name ablation_eqcap_jnu
```

### 4.4 OOD score comparison for new baselines only (no retraining)

```bash
python scripts/compare_ood_scores_from_checkpoints.py \
  --outputs_root outputs_scenarios \
  --methods AFRDN,WPA_PDG \
  --exp_prefixes tii319_afrdn_wpa_c2j,tii319_afrdn_wpa_j2c \
  --score_types msp,energy,maxlogit \
  --energy_temp 1.0 \
  --unknowns 1,2,3 \
  --seeds 41,42,43 \
  --settings B,C \
  --out_dir outputs_scenarios/ood_score_compare_afrdn_wpa
```

### 4.5 Pseudo-environment alignment analysis and figures

```bash
bash scripts/run_pseudoenv_alignment.sh \
  --mode all \
  --unknowns 1,2,3 \
  --seeds 41,42,43 \
  --out_tag pseudoenv_alignment_all_u123_s414243
```

## 5. Key Outputs

- Scenario summary:
  - `outputs_scenarios/<exp_name>/summary.csv`
  - `outputs_scenarios/<exp_name>/summary.md`
- Multi-run aggregation:
  - `outputs_scenarios/<exp_prefix>/summary_mean_pm_std_by_case.csv`
  - `outputs_scenarios/<exp_prefix>/summary_mean_pm_std_overall.csv`
- Ablation summary:
  - `outputs_ablation/<exp_name>/summary.csv`
- Pseudo-env alignment analysis:
  - `outputs_scenarios/<tag>_analysis/alignment_unknown_rotation_overall.csv`
  - `outputs_scenarios/<tag>_analysis/heatmaps/*.png`
  - `outputs_scenarios/<tag>_analysis/sankey/*.png`

## 6. Documentation Index (Chinese)

- Baseline addition (AFRDN + WPA-PDG): `BASELINE_AFRDN_WPAPDG_README_CN.md`
- Equal-capacity ablation: `ABLATION_EQCAP_README_CN.md`
- OOD score protocol (MSP/Energy/MaxLogit): `OOD_README_CN.md`
- Pseudo-environment interpretability: `PSEUDOENV_ALIGNMENT_README_CN.md`
- SELF-only workflow: `SELF_EXPERIMENT_GUIDE_CN.md`

## 7. Reproducibility Notes

- Use fixed seeds `41,42,43` and unknown rotations `1,2,3`.
- Keep evaluation protocol unified across methods:
  - same split and known/unknown rotation,
  - same epochs/batch size/input length/normalization,
  - same OOD score rule when comparing methods.
- For paper tables, report `KnownAcc`, `AUROC`, `AUPR`, `FPR95` under Setting B/C.

## 8. Citation

A formal citation entry will be updated after manuscript metadata is finalized.
For now, please cite the corresponding TII submission title used in your manuscript.

## 9. License

This is a pre-release research codebase for academic evaluation.
Formal open-source licensing terms will be added after manuscript process finalization.
