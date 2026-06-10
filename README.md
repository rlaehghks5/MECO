# MeCo: One-Step MeanFlow-based Corrector for Multi-Channel Speech Separation
[![PWC](https://img.shields.io/badge/INTERSPEECH-paper-red)](https://arxiv.org/abs/2606.09677) 


> Official code repository for **MeCo**, accepted to **Interspeech 2026**.

MeCo is a one-step generative **corrector** for multi-channel joint speech separation,
denoising, and dereverberation. Given the imperfect estimate of a discriminative
separator, MeCo learns a **conditional average velocity field** (Mean Flows) that maps
the discriminative output directly onto the clean speech manifold in a **single forward
pass (1 NFE)** — no iterative ODE/SDE solving, no two-stage distillation, and no
heuristic trajectory truncation.

To maximize one-step performance, MeCo is trained with **Data-Space Optimization (DSO)**,
which combines:

- an **xr-loss** that scales the velocity-matching loss by the squared displacement
  interval (∆²), serving as a generative objective for natural human-listening quality
- an **Endpoint SI-SDR loss** that directly optimizes terminal signal fidelity by
  simulating the actual one-step inference during training.

MeCo achieves SOTA performance with minimal computational overhead (+1 NFE, +0.0068 RTF),
delivering high signal fidelity **and** superior listening quality on both in-domain and
out-of-domain scenarios.

---

## 📊 Results (Single-channel)
While our paper focuses on the multi-channel setting, MeCo also generalizes to the
**single-channel** scenario. We use the same setup as in the Interspeech paper, but feed
only the **reference channel** to the separator and corrector. Concretely, the evaluation
uses the reference channel of the same dataset (WSJ0 + DEMAND, 16 kHz, containing both
reverberation and noise, with 1-, 2-, and 3-speaker mixtures). Using CrossNet as the
discriminative separator, MeCo consistently outperforms both the baseline separator and
the FastGeCo corrector across all signal-fidelity and listening-quality metrics,
demonstrating that the proposed corrector is not specific to the multi-channel case.

|  | PESQ_WB | ESTOI | Si-SDR | NISQA  | DNSMOS | UTMOS |
| --- | --- | --- | --- | --- | --- | --- |
| CrossNet  | 1.45 | 0.56 | 3.72 | 3.03 | 2.53 | 2.21 |
| CrossNet + FastGeCo | 1.50 | 0.61 | 4.42 | 3.35 | 2.75 | 2.60 |
| CrossNet + MeCo (Ours) | **1.51** | **0.63** | **4.65** | **3.72** | **2.85** | **2.88** |

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/rlaehghks5/MECO.git
cd MECO
pip install -r requirements.txt
```

---


## 🚀 Training

### Run with the provided script

```bash
bash flowmse/scripts/train_multi-channel_one-step_MeCo.sh
```

## 🧪 Evaluation (One-Step Inference)

### Run with the provided script

```bash
bash eval_meanflow.sh
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{meco2026,
  author    = {Dohwan Kim and Jung-Woo Choi},
  title     = {{MeCo}: One-Step {MeanFlow}-based Corrector for Multi-Channel Speech Separation},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```

---

## 📚 Built Upon & Related Work
- **FLOWMSE (FlowSE)** — https://github.com/seongq/flowmse
- **MEANFLOWSE (MeanFLowSE)** — https://github.com/liduojia1/MeanFlowSE
