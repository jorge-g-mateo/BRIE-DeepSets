# BRIE-DeepSets: Scalable Set-Based User Modeling for Visual Explainability in Recommender Systems

**Authors:**
Jorge García-Mateo<sup>1</sup>, Jorge Paz-Ruza<sup>2</sup>, Carlos Eiras-Franco<sup>2</sup>

<sup>1</sup> Facultade de Informática, Universidade da Coruña, A Coruña, Spain 

<sup>2</sup> Universidade da Coruña, Grupo LIDIA - CITIC, A Coruña, Spain

---

## 1. Overview

This repository presents **BRIE-DeepSets**, an extension of BRIE for visual explainability in recommender systems.

Visual explainability aims to justify recommendations using images that are relevant to each user. Existing models such as ELVis, MF-ELVis and BRIE represent users through learned ID-based embeddings. Although effective, this design makes the number of trainable parameters grow with the number of users and prevents the direct representation of users not seen during training.

BRIE-DeepSets replaces user-ID embeddings with a **set-based user representation** computed from the visual embeddings of the images in each user's history.

The proposed approach:

* removes the user embedding table,
* keeps the number of trainable parameters constant across datasets,
* naturally supports unseen users,
* preserves the Bayesian Personalized Ranking objective used in BRIE,
* maintains competitive ranking performance while greatly improving scalability.

The main trade-off is that user representations are computed dynamically from visual histories, which increases computational cost compared with directly looking up a stored user embedding.

---

## 2. Method

Each user is represented as a set of image embeddings:

$$
X_u = {x_1, x_2, ..., x_{N_u}}
$$

The user embedding is computed using a Deep Sets encoder:

$$
u = \rho \left( \sum_{i=1}^{N_u} \phi(x_i) \right)
$$

where:

* $x_i$ is the embedding of an image associated with the user,
* $\phi(\cdot)$ is a learnable transformation applied independently to each image,
* the sum aggregation makes the representation permutation-invariant,
* $\rho(\cdot)$ maps the aggregated vector to the final user embedding.

This formulation makes the user representation independent of the image order and removes the need for user-specific trainable parameters.

The model keeps the BRIE pairwise ranking formulation. Given a user embedding $u$, a positive image $i^+$ and a negative image $i^-$, the score is computed as:

$$
\hat{y}_{u,i} = u^\top v_i
$$

and the BPR loss is:

$$
\mathcal{L}*{BPR} = - \log \sigma \left( \hat{y}*{u,i^+} - \hat{y}_{u,i^-} \right)
$$

---

## 3. Key Contributions

* Deep Sets–based user representation for BRIE.
* Removal of ID-based user embedding tables.
* Constant number of trainable parameters across datasets.
* Natural support for user cold-start scenarios.
* Evaluation on six real-world restaurant recommendation datasets.
* Analysis of the trade-off between ranking performance, scalability and computational cost.

---

## 4. Experimental Setup

Experiments were conducted on six city datasets:

* **Gijón**
* **Barcelona**
* **Madrid**
* **New York**
* **Paris**
* **London**

For parameter counting, the latent dimensions from the original models are preserved:

| Model         | Latent dimension |
| ------------- | ---------------: |
| ELVis         |              256 |
| MF-ELVis      |             1024 |
| BRIE          |               64 |
| BRIE-DeepSets |               64 |

---

## 5. Results Summary

BRIE obtains the best ranking results overall, but BRIE-DeepSets achieves competitive performance with a much smaller and constant number of trainable parameters.

| City | BRIE Params | BRIE MAUC | BRIE-DeepSets Params | BRIE-DeepSets MAUC | Parameter Reduction |
|---|---:|---:|---:|---:|---:|
| Gijón | 427,264 | 0.643 | 209,472 | 0.635 | **51.0%** |
| Barcelona | 2,244,736 | 0.663 | 209,472 | 0.658 | **90.7%** |
| Madrid | 2,940,288 | 0.673 | 209,472 | 0.668 | **92.9%** |
| New York | 3,369,344 | 0.677 | 209,472 | 0.672 | **93.8%** |
| Paris | 3,055,936 | 0.666 | 209,472 | 0.661 | **93.1%** |
| London | 8,726,592 | 0.665 | 209,472 | 0.663 | **97.6%** |

BRIE-DeepSets obtains the second-best ranking performance in all six datasets. The differences with respect to BRIE are small, especially in MAUC, while the parameter reduction is substantial.

In London, for example, MAUC decreases only from **0.665** to **0.663**, while the number of trainable parameters is reduced from **8,726,592** to **209,472**, corresponding to a **97.6% reduction**.

The reduction is even larger when compared with ELVis and MF-ELVis, mainly because these models use larger latent dimensions in their original configurations.

The BRIE-DeepSets configuration was selected through a limited random search on Barcelona and then reused across the remaining cities without city-specific retuning. Therefore, the results should not be interpreted as an exhaustive optimization for each dataset.

---

## 6. Installation

```bash
pip install -r requirements.txt
```

Recommended environment:

* Python ≥ 3.10
* PyTorch
* PyTorch Lightning
* CUDA-enabled GPU

---

## 7. Usage

### Training BRIE-DeepSets

```bash
python main.py \
  --stage train \
  --city CITY_NAME \
  --model BRIE_DEEPSETS \
  --max_epochs EPOCHS \
  --batch_size BATCH_SIZE \
  --lr LR \
  --dropout DROPOUT \
  -d LATENT_DIM \
  --workers NUM_WORKERS
```

### Evaluating BRIE-DeepSets

```bash
python main.py \
  --stage test \
  --city CITY_NAME \
  --model BRIE_DEEPSETS \
  --batch_size BATCH_SIZE \
  --workers NUM_WORKERS
```

### Training or evaluating the original BRIE baseline

Use:

```bash
--model BRIE
```

To see all available command-line options:

```bash
python main.py --help
```

---

## 8. Main Takeaways

* BRIE remains the strongest model in pure ranking performance.
* BRIE-DeepSets provides a scalable alternative with competitive ranking results.
* The number of trainable parameters remains constant at **209,472** across all datasets.
* The model naturally supports unseen users because it does not rely on user-ID embeddings.
* The main cost of the proposed approach is the dynamic aggregation of each user's visual history.

Overall, BRIE-DeepSets is useful when scalability, parameter efficiency and cold-start capability are more important than achieving the maximum possible ranking score with user-specific embeddings.

---

## 9. Relationship to BRIE

This repository extends the original BRIE model and reuses its recommendation and visual explainability framework.

All credit for the original BRIE architecture, training strategy and evaluation protocol belongs to the authors of the original work.

Original repository:
https://github.com/Kominaru/BRIE

---

## 10. License

This project follows the same license as the original BRIE repository.
