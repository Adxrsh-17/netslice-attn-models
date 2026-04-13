# slicecast

**Slice-level 5G network traffic forecasting from KPI streams using a hybrid VAR–GRU–Transformer framework.**

`slicecast` converts raw 5G PCAP captures into multivariate KPI time-series and applies a three-stage hybrid deep learning pipeline to forecast traffic load, latency, and congestion — separately for each network slice type (eMBB, URLLC, mMTC).


---

## Background & Motivation

Modern 5G networks generate high-frequency, non-stationary traffic streams across service slices with fundamentally different characteristics:

- **eMBB** — high-bandwidth, macro-burst patterns (video streaming, broadband)
- **URLLC** — sparse, stochastic micro-bursts with strict latency constraints (autonomous systems, critical IoT)
- **mMTC** — periodic, low-volume IoT reporting intervals (sensor networks)

Existing 5G analytics frameworks — including NWDAF-based systems — predict **aggregate** network traffic and do not provide per-slice forecasting. This project fills that gap by building a slice-aware forecasting pipeline that learns each slice's unique temporal dynamics.

---

## Research Gap

Prior work (Di Mauro et al. 2024, Zhang & Tang 2024, Azalim & Silva 2024) either:
- Forecasts aggregate 4G/5G traffic without slice disaggregation
- Applies ML within NWDAF for classification, not time-series forecasting
- Uses a single "universal" model architecture across all traffic types

`slicecast` proves that a single architecture cannot optimally serve all three slice types and proposes **dynamic slice-aware model orchestration** instead.

---

## Proposed Solution

A novel **VAR–GRU–Transformer hybrid framework** trained and evaluated separately for each slice:

```
Stage 1 — VAR Baseline
  Captures long-range linear multivariate cross-KPI correlations

Stage 2 — GRU Residual Corrector
  Processes VAR residuals to model short-term non-linear bursts

Stage 3 — Transformer Fusion Head (slice-dependent)
  Routes GRU output into one of four advanced architectures:
    ├── N-BEATS      (basis expansion, general forecasting)
    ├── PatchTST     (patch-based tokenization, eMBB)
    ├── TimeMixer    (multi-scale temporal mixing, mMTC)
    └── TFT          (temporal fusion transformer, URLLC)
```

The optimal head is selected per slice through benchmarking, rather than fixed globally.

---

## Dataset

**5G Traffic Capture Dataset: User Equipment Classification in the 5G Core**

| Field | Details |
|---|---|
| Source | [Zenodo — DOI: 10.5281/zenodo.15064130](https://zenodo.org/records/15064130) |
| Creators | Leonardo Azalim de Oliveira et al., Universidade Federal de Juiz de Fora |
| Environment | Simulated using free5GC + UERANSIM |
| Capture tool | tcpdump (raw PCAP, full packet headers + payloads) |
| License | CC-BY 4.0 |
| Files | `Youtube_cellular.pcap` (12.4 GB), `naver5g3-10M.pcap` (11.8 GB), `training_data_mmtc.zip` (18.7 MB) |

---

## Pipeline

### Phase 1 — Distributed Data Ingestion
- Raw PCAP files (20 GB+) stored on HDFS
- Custom Scala `PcapSharder` splits files into 1 GB chunks, preserving 24-byte global PCAP headers

### Phase 2 — Native Decoding & Feature Engineering
- Spark Structured Streaming reads PCAP shards continuously
- `PcapKpiExtractor` decodes packets using a 1-second sliding window
- **36 base KPIs** extracted (Throughput, Jitter, Packet Sizes, Latency, etc.)
- Rolling window (length = 5) applied: means, standard deviations, first-order differentials
- Expanded to a **44-dimensional feature space**, exported as Parquet to HDFS

### Phase 3 — Hybrid Predictive Modeling
- Parquet data dimensionally reduced to **7 core forecasting KPIs**
- 80/20 chronological train/test split — no data leakage
- Three-stage VAR–GRU–Transformer hybrid trained in a GPU-accelerated Kaggle environment
- Four Transformer heads trained per slice for comparative benchmarking

### Phase 4 — Evaluation & Benchmarking
- 10-step ahead forecasting window
- Evaluated on RMSE, MAE, and R² across all 7 KPIs per slice
- Post-training diagnostics: residual distributions, CUSUM stability charts, radar charts
- Cross-slice comparative analysis to identify the optimal architecture per slice type

---

## Models

### Model 1 — VAR(3) + GRU + N-BEATS
Input `(B, 60, 7)` → GRU (128 units, drop=0.2) → 3 N-BEATS blocks with residual connections → Σ forecast `(B, 7)` — trained with Huber loss (δ=1), Adam 1e-4

### Model 2 — VAR(3) + GRU + PatchTST
Input → GRU (64 units) → Conv1D patch tokenization (k=12, s=6, 9 patches) → Positional embedding → 2× Transformer encoder (MHA h=4, d_k=16) → GAP → Dense head `(B, 7)`

### Model 3 — VAR(3) + GRU + TimeMixer
Input → GRU (64 units, return_seq=True) → 2× TimeMixer blocks at 3 temporal scales (full@60, half@30, quarter@15) with avg-pool + upsample fusion → Flatten → Dense head `(B, 7)`

### Model 4 — VAR(3) + GRU + TFT
Input → GRN₁ VarSel (ELU + GLU gating) → GRU₁ (128 units) → GRU₂ (64 units) → MHA (heads=4, d_k=32) → LN+Res → GAP → GRN₂ Out → Dense7 linear `(B, 7)`

---

## Evaluation Metrics

| Metric | Role |
|---|---|
| Huber Loss (δ=1) | Training objective — robust to micro-burst outliers, prevents gradient instability |
| MAE | Primary evaluation metric |
| RMSE | Sensitivity to large prediction errors |
| R² | Post-training diagnostic — used to expose model hallucinations on stochastic traffic (e.g., negative R² on URLLC throughput correctly identifies un-trendable noise) |

> R² is intentionally **not** used during training — it penalizes small time-shifts common in 1-second 5G burst traffic.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch / TensorFlow | Model implementation |
| Apache Spark (Structured Streaming) | Distributed PCAP ingestion and KPI computation |
| Scala | Custom PcapSharder for HDFS chunking |
| Hadoop HDFS | Distributed storage for raw and processed data |
| Kaggle (GPU) | Model training environment |
| Pandas / NumPy | Feature engineering |
| Matplotlib / Seaborn | Result visualization |
| ONNX | Model export for real-time inference |

---

## Project Structure

```
slicecast/
├── run_pipeline.sh               # Master orchestration script
├── requirements.txt
├── config/
│   └── model_config.yaml         # Hyperparameters per slice and model
├── ingestion/
│   ├── PcapSharder.scala         # HDFS PCAP sharding
│   └── PcapKpiExtractor.py       # Spark streaming KPI extractor
├── features/
│   └── temporal_features.py      # Rolling window feature expansion
├── models/
│   ├── var_baseline.py           # VAR(3) baseline
│   ├── gru_corrector.py          # GRU residual corrector
│   ├── nbeats_head.py            # N-BEATS fusion head
│   ├── patchtst_head.py          # PatchTST fusion head
│   ├── timemixer_head.py         # TimeMixer fusion head
│   └── tft_head.py               # Temporal Fusion Transformer head
├── training/
│   ├── slice_trainer.py          # Slice-aware training loop
│   └── benchmarker.py            # Cross-model evaluation
├── evaluation/
│   ├── metrics.py                # RMSE, MAE, R² computation
│   └── diagnostics.py            # CUSUM, residual, radar charts
├── inference/
│   └── spark_inference.py        # Real-time Spark streaming inference
└── outputs/
    ├── models/                   # ONNX + scaler checkpoints
    ├── plots/                    # Forecast and diagnostic plots
    └── logs/                     # Training logs per slice
```

---

## Setup

**1. Clone and set up environment**

```bash
git clone https://github.com/Adxrsh-17/slicecast.git
cd slicecast
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Download the dataset**

```bash
# Download from Zenodo (DOI: 10.5281/zenodo.15064130)
wget https://zenodo.org/records/15064130/files/training_data.zip
unzip training_data.zip -d data/raw/
```

**3. Set up Hadoop/Spark** (required for ingestion phase)

Ensure `HADOOP_HOME` and `SPARK_HOME` are configured. HDFS must be running locally or remotely.

---

## Usage

**Run the full pipeline:**

```bash
bash run_pipeline.sh
```

**Run ingestion and KPI extraction only:**

```bash
python ingestion/PcapKpiExtractor.py --input hdfs:///data/raw/ --output hdfs:///data/kpi/
```

**Train models for a specific slice:**

```bash
python training/slice_trainer.py --slice embb --model patchtst
python training/slice_trainer.py --slice urllc --model tft
python training/slice_trainer.py --slice mmtc --model timemixer
```

**Run benchmarking across all heads:**

```bash
python training/benchmarker.py --slice embb
```

**Run real-time inference:**

```bash
python inference/spark_inference.py --model outputs/models/embb_patchtst.onnx
```

---

## Comparative Study

| Aspect | Di Mauro et al. (2024) | slicecast |
|---|---|---|
| Architecture | 2-stage: VAR + CNN/LSTM/GRU | 3-stage: VAR + GRU + Transformer head |
| Domain | Aggregated 4G VoIP traffic | Isolated 5G slices (raw PCAP) |
| Model strategy | Single universal architecture | Slice-aware dynamic head selection |
| Loss function | MAE / RMSE | Huber Loss (outlier-robust) |
| Evaluation | MAE, RMSE | MAE, RMSE, R² (with diagnostic use) |

---

## Novelty

- First framework to perform **per-slice** (eMBB, URLLC, mMTC) traffic forecasting from raw 5G PCAP data — not aggregate load prediction
- Extends state-of-the-art VAR-GRU baselines with Transformer fusion heads
- Mathematically invalidates the "one-size-fits-all" model assumption for 5G slices
- Full PCAP-to-inference pipeline with distributed ingestion via Apache Spark on HDFS

---

## Limitations

- Dataset sourced from a simulated 5G environment (free5GC + UERANSIM), not a live commercial network
- Results are bounded by the traffic scenarios and slice configurations present in the dataset

---



---


---

## License

MIT License — see [LICENSE](LICENSE) for details.
