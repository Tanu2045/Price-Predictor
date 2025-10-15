# DistilBERT Price Prediction Baseline

## Training and Validation Strategy
- The training CSV is filtered for non-empty catalog entries and split into train/validation subsets with `train_test_split` using a fixed project seed.
- Prices are discretised into up to 20 quantile bins (`pd.qcut`) to approximate stratified sampling, preserving the price distribution inside a regression setup.
- Sample identifiers for each split are persisted to `outputs/splits/bert_split.json` to keep evaluation and inference runs aligned.

## Tokenisation and Data Loading
- Text is tokenised on the fly with `AutoTokenizer.from_pretrained` for DistilBERT, using dynamic padding, truncation to the configured `max_length`, and batched tensor outputs compatible with `accelerate`.
- `TextPriceDataset` exposes `sample_id`, raw text, and price targets; the collate function appends `labels` tensors and keeps hooks for future tabular side inputs.
- Data loaders honour the configured batch size, gradient accumulation steps, and optional worker threads while pinning memory when CUDA is available.

## Model, Optimisation, and Metrics
- `BertRegressor` wraps a DistilBERT backbone with a lightweight dropout → linear → GELU → dropout → linear regression head that predicts a single price scalar per sample.
- Training minimises mean absolute error (`nn.L1Loss`) under AdamW with configurable weight decay, warmup ratio, cosine or linear scheduling, gradient clipping, and mixed precision via `accelerate`.
- Validation tracks MAE and SMAPE; improvements in SMAPE trigger checkpoint exports to `checkpoints/bert/best` alongside the metric manifest for reproducibility.
- Early stopping monitors SMAPE with a configurable patience window to halt runs that stop improving.

## Tech Stack
- Python tooling managed through `uv`, with core dependencies on PyTorch, Hugging Face Transformers, Accelerate, scikit-learn, pandas, NumPy, and tqdm.
- Configuration is centralised in YAML (`config/paths.yml`) and hydrated through custom loaders that resolve dataset paths relative to the project root.

## Rationale for DistilBERT
- DistilBERT was selected as the initial encoder because it delivers competitive accuracy on catalogue text with modest GPU memory requirements and fast iteration cycles.
- The regression head leverages the CLS token representation, enabling fine-tuning without the cost of larger encoder variants while still capturing contextual product descriptors.

## Future Plans and Ensemble Groundwork
- `master.ensemble.average_prediction_files` already supports averaging `sample_id`-aligned CSVs, enabling straightforward blending of BERT, vision, and planned gradient-boosted models.
- The dataset module exposes `placeholder_feature_harness` to ingest cached numerical features, providing the conduit for regex-derived attributes and cross-modal embeddings.
- A forthcoming pipeline will reuse BERT (or alternative encoders) to generate document embeddings that feed an XGBoost regressor, expanding the ensemble beyond neural models.
- Vision backbones (ViT/EfficientNet) and tabular feature extractors will plug into the same logging, checkpointing, and validation scaffolding established for the DistilBERT baseline.
