# CWSA: Confidence-Weighted Selective Accuracy

This repository contains the official implementation of two novel evaluation metrics for selective prediction and trust-aware classification:

- **CWSA** – Confidence-Weighted Selective Accuracy  
- **CWSA+** – Normalized Confidence-Weighted Selective Accuracy

These metrics are designed to reward confident correctness and penalize overconfident errors, providing a deeper understanding of model reliability across confidence thresholds.

---

## 🔧 Project Structure

```
.
├── mnist_cnn.py               # Train and evaluate CNN on MNIST
├── cifar_cnn.py               # Train and evaluate CNN on CIFAR-10
├── simulate_models.py         # Synthetic models (calibrated, overconfident, etc.)
├── stress_noise.py            # Label noise robustness
├── stress_shift.py            # Confidence degradation test
├── stress_imbalance.py        # Class imbalance test
├── Metrics/                   # Implementation of CWSA and CWSA+
├── Experiments/               # Evaluation runners and sweeps
├── Results/                   # Output CSVs for metrics and plots
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT license
```

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/your-username/cwsa-eval.git
cd cwsa-eval
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run synthetic model evaluation
```bash
python simulate_models.py
```

### Train and evaluate on real datasets
```bash
python mnist_cnn.py
python cifar_cnn.py
```

---

## 📊 Metric Comparison

| Metric              | Penalizes Overconfidence | Normalized | Threshold-local | Deployment-friendly |
|---------------------|---------------------------|------------|------------------|----------------------|
| Accuracy            | ❌                        | ✅         | ❌               | ✅                   |
| Selective Accuracy  | ❌                        | ✅         | ✅               | ✅                   |
| ECE                 | ✅ (aggregate)            | ✅         | ❌               | ❌                   |
| AURC                | ✅ (global risk)          | ✅         | ❌               | ✅                   |
| **CWSA**            | ✅ **(per prediction)**   | ❌         | ✅               | ✅✅✅               |
| **CWSA+**           | ✅ **(confidence-rewarded)** | ✅      | ✅               | ✅✅✅               |

---

## 📁 Results Overview

All outputs are saved under `/Results` as `.csv` files across:

- ✅ Real-world datasets: MNIST, CIFAR-10  
- 🧪 Synthetic Models: Calibrated, Overconfident, Underconfident, Perfect, Random  
- ⚠️ Stress Tests: Noise injection, Confidence degradation, Class imbalance

Each file contains sweeps over confidence thresholds for:
- Selective Accuracy
- CWSA
- CWSA+
- ECE
- AURC
- Coverage

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{cwsa2024,

}
```

---

## 🔍 License

This repository is licensed under the [MIT License](LICENSE).

---

## ❤️ Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 requirements.txt

```
numpy
pandas
scikit-learn
matplotlib
scipy
tqdm
```

---

## 📄 LICENSE (MIT License)

```
MIT License

Copyright (c) 2024 Kourosh Shahnazari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
