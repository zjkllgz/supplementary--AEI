# supplementary--AEI

> **Important Note on Data Confidentiality**  
> The original real-world industrial dataset is confidential and cannot be publicly released.  
> This repository provides `all_data.pkl`, a **sanitized and normalized** version of the data, which is used to reproduce the proposed method pipeline and code execution.

---

## 1. Contents

- `phmoea_real.py`  
  Implementation of PHMOEA on the sanitized real-world industrial dataset. It calls the MS–BCNN backbone defined in `ms_bcnn.py`.

- `phmoea_synthetic_benchmarks.py`  
  Implementation of PHMOEA on hierarchical synthetic benchmark problems.

- `ms_bcnn.py`  
  MS–BCNN model, dataset preparation, training, and evaluation utilities.

- `all_data.pkl`  
  Normalized and sanitized data used for running the real-world experimental pipeline. The confidential raw data are not included.

---

## 2. Environment

### 2.1 Python

- Python 3.9 or higher is recommended.
- Tested on macOS and Linux.

### 2.2 Dependencies

Please install the following common scientific computing packages:

- numpy
- scipy
- pandas
- scikit-learn
- torch
- tqdm

Thank you for reviewing our supplementary materials.
