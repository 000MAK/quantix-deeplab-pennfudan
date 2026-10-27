# DeepLabV3 Pedestrian Segmentation | Penn-Fudan

A small torchvision DeepLabV3-MobileNetV3 model trained from scratch for binary pedestrian-mask segmentation. The script downloads the Penn-Fudan data, creates deterministic train/validation/test splits, and saves a held-out pixel-accuracy summary.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --epochs 1
```

See `python main.py --help` for options. Dataset files are cached in `data/`; checkpoints and metrics are written to `outputs/`. These runtime artifacts are excluded from Git.

## Data and limits

Source: [University of Pennsylvania - Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/). It is a small educational dataset; one run is a pipeline demonstration, not a meaningful general-purpose benchmark.

## Results

A one-epoch local run reached held-out test **pixel accuracy 0.854**. Background dominates the pixels, so pixel accuracy alone overstates quality; report pedestrian IoU before interpreting the model.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
