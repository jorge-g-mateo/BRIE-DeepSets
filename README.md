# BRIE-DeepSets: Scalable Set-Based User Modeling for Visual Explainability in Recommender Systems

**Authors:**
Jorge García-Mateo<sup>1</sup>, Jorge Paz-Ruza<sup>2</sup>, Carlos Eiras-Franco<sup>2</sup>

<sup>1</sup> Facultade de Informática, Universidade da Coruña, A Coruña, Spain <sup>2</sup> Universidade da Coruña, Grupo LIDIA - CITIC, A Coruña, Spain

**Contact:**
[jorge.garcia.mateo@udc.es](mailto:jorge.garcia.mateo@udc.es)
[j.ruza@udc.es](mailto:j.ruza@udc.es), [carlos.eiras.franco@udc.es](mailto:carlos.eiras.franco@udc.es)

---

## 1. Overview

This repository presents **BRIE-DeepSets**, an extension of the BRIE model for visual explainability in recommender systems.

The main idea is to replace the traditional user representation based on fixed user identifiers with a **set-based user representation computed from the visual embeddings of the images associated with each user**.

In the original BRIE model, each user is represented by a learned embedding indexed by a unique user identifier. This design is effective, but it has several scalability limitations:

* The number of trainable parameters grows linearly with the number of users.
* New users require new user-specific embeddings.
* The model does not naturally handle user cold-start scenarios.
* Large datasets lead to increasingly large embedding matrices.

BRIE-DeepSets addresses these limitations by representing each user as a permutation-invariant set of image embeddings. The user representation is computed dynamically from the content of the user’s image set, instead of being stored as a dedicated parameter vector.

As a result:

* The number of trainable parameters becomes **constant across datasets**.
* The model no longer requires user-specific embedding tables.
* New users can be represented from their available images without changing the architecture.
* The approach improves scalability while preserving the ranking and explainability-oriented structure of BRIE.

The proposed method keeps the Bayesian Pairwise Ranking objective used in BRIE and integrates the Deep Sets user encoder within the original training and evaluation pipeline.

---

## 2. Key Contributions

* Replacement of fixed user-ID embeddings with **Deep Sets–based user representations**.
* Elimination of user-specific trainable parameters.
* Constant model size regardless of the number of users.
* Natural support for user cold-start scenarios.
* Seamless integration with the BRIE training and evaluation pipeline.
* Empirical evaluation on six real-world city datasets.
* Explicit analysis of the trade-off between scalability and ranking performance.

---

## 3. Architecture

### 3.1 User Representation with Deep Sets

Each user is represented as a set of visual embeddings extracted from images associated with that user.

The user embedding is computed following the Deep Sets formulation:

$$
u = \rho \left( \sum_{i=1}^{N_u} \phi(x_i) \right)
$$

where:

* $x_i$ is the embedding of the $i$-th image associated with the user,
* $\phi(\cdot)$ is a learnable transformation applied independently to each image embedding,
* the aggregation operator is permutation-invariant,
* $\rho(\cdot)$ maps the aggregated representation to the final user embedding.

This formulation guarantees that the resulting user representation does not depend on the order of the images in the set.

Importantly, the user embedding is computed dynamically from image content and does not rely on stored user-specific parameters.

---

### 3.2 Deep Sets Block in BRIE

The Deep Sets block is composed of two neural components:

* **$\phi$ network**: an MLP applied independently to each image embedding.
* **$\rho$ network**: an MLP applied after the aggregation step.

The aggregation is implemented using sum pooling:

$$
z_u = \sum_{i=1}^{N_u} \phi(x_i)
$$

$$
u = \rho(z_u)
$$

This design provides:

* Permutation invariance.
* Support for variable-sized user histories.
* Independence from the number of users.
* Constant parameter count across datasets.

As a consequence, the number of trainable parameters does not grow with the user base.

---

### 3.3 Ranking Objective

The model preserves the Bayesian Pairwise Ranking objective used in BRIE.

Given a training triplet:

$$
(u, i^+, i^-)
$$

where:

* $u$ is the user embedding obtained through the Deep Sets encoder,
* $i^+$ is a positive image associated with the user,
* $i^-$ is a negatively sampled image,

the preference score is computed as:

$$
\hat{y}_{u,i} = u^\top v_i
$$

where $v_i$ is the latent embedding of image $i$.

The BPR loss is defined as:

$$
\mathcal{L}*{BPR} = - \log \sigma \left( \hat{y}*{u,i^+} - \hat{y}_{u,i^-} \right)
$$

This objective encourages the model to rank positive images above negative ones while maintaining the explainability-oriented structure of BRIE.

---

## 4. Experimental Setup

### 4.1 Environment

* Python ≥ 3.10
* PyTorch
* PyTorch Lightning
* CUDA-enabled GPU recommended

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

### 4.2 Datasets

Experiments were conducted on six real-world image-based recommendation datasets:

* **Gijón**
* **Barcelona**
* **Madrid**
* **New York**
* **Paris**
* **London**

Datasets follow the same structure as the original BRIE repository and should be placed under the `data/` directory.

---

### 4.3 Parameter Counting

For a fair comparison with the original baselines, the latent dimensionalities used in the original works are respected when computing the number of trainable parameters:

* **ELVis**: $d = 256$
* **MF-ELVis**: $d = 1024$
* **BRIE**: $d = 64$
* **BRIE-DeepSets**: $d = 64$

This is important because ELVis and MF-ELVis use larger latent dimensions, which increases their number of parameters substantially.

---

## 5. Usage

### 5.1 Training

To train BRIE-DeepSets:

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

To train the original BRIE baseline, use:

```bash
python main.py \
  --stage train \
  --city CITY_NAME \
  --model BRIE \
  --max_epochs EPOCHS \
  --batch_size BATCH_SIZE \
  --lr LR \
  --dropout DROPOUT \
  -d LATENT_DIM \
  --workers NUM_WORKERS
```

---

### Common Training Options

| Argument       | Description                                     |
| -------------- | ----------------------------------------------- |
| `--city`       | Dataset / city name, e.g. `barcelona`, `madrid` |
| `--stage`      | Execution stage: `train` or `test`              |
| `--model`      | Model name, e.g. `BRIE`, `BRIE_DEEPSETS`        |
| `--batch_size` | Batch size                                      |
| `--max_epochs` | Number of training epochs                       |
| `--lr`         | Learning rate                                   |
| `-d`           | Latent embedding dimensionality                 |
| `--dropout`    | Dropout rate                                    |
| `--workers`    | Number of dataloader workers                    |
| `--seed`       | Random seed for reproducibility                 |

---

### Optional Training Flags

| Flag               | Description                                  |
| ------------------ | -------------------------------------------- |
| `--early_stopping` | Enable early stopping                        |
| `--no_validation`  | Disable validation split                     |
| `--use_train_val`  | Train using both train and validation splits |
| `--ckpt_path`      | Path to a checkpoint to resume training      |

---

### Deep Sets–Specific Options

| Flag                | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `--max_user_images` | Maximum number of images used to build each user set |

---

### 5.2 Evaluation

To evaluate BRIE-DeepSets:

```bash
python main.py \
  --stage test \
  --city CITY_NAME \
  --model BRIE_DEEPSETS \
  --batch_size BATCH_SIZE \
  --workers NUM_WORKERS
```

To evaluate BRIE:

```bash
python main.py \
  --stage test \
  --city CITY_NAME \
  --model BRIE \
  --batch_size BATCH_SIZE \
  --workers NUM_WORKERS
```

---

### 5.3 Listing All Available Options

To see the full and up-to-date list of command-line options and default values, run:

```bash
python main.py --help
```

---

## 6. Results

The main objective of BRIE-DeepSets is not to obtain marginal improvements in ranking performance, but to improve **scalability** by removing the dependency between the number of model parameters and the number of users.

The following tables report ranking performance and the number of trainable parameters for all evaluated cities.

Best results are shown in **bold**. Second-best results are shown with <ins>underline</ins>.

---

### 6.1 Gijón

| Model         |      Params |       MRecall@10 |         MNDCG@10 |             MAUC |
| ------------- | ----------: | ---------------: | ---------------: | ---------------: |
| RND           |           — |            0.373 |            0.185 |            0.487 |
| CNT           |           — |            0.464 |            0.218 |            0.546 |
| ELVis         |   1,840,641 |            0.521 |            0.262 |            0.596 |
| MF-ELVis      |   6,836,224 |            0.538 |            0.285 |            0.592 |
| BRIE          |     427,264 |        **0.607** |        **0.333** |        **0.643** |
| BRIE-DeepSets | **209,472** | <ins>0.571</ins> | <ins>0.303</ins> | <ins>0.635</ins> |

---

### 6.2 Barcelona

| Model         |      Params |       MRecall@10 |         MNDCG@10 |             MAUC |
| ------------- | ----------: | ---------------: | ---------------: | ---------------: |
| RND           |           — |            0.409 |            0.186 |            0.502 |
| CNT           |           — |            0.443 |            0.219 |            0.554 |
| ELVis         |   9,110,529 |            0.597 |            0.327 |            0.631 |
| MF-ELVis      |  35,915,776 |            0.557 |            0.293 |            0.596 |
| BRIE          |   2,244,736 |        **0.630** |        **0.368** |        **0.663** |
| BRIE-DeepSets | **209,472** | <ins>0.610</ins> | <ins>0.343</ins> | <ins>0.658</ins> |

---

### 6.3 Madrid

| Model         |      Params |       MRecall@10 |         MNDCG@10 |             MAUC |
| ------------- | ----------: | ---------------: | ---------------: | ---------------: |
| RND           |           — |            0.374 |            0.171 |            0.499 |
| CNT           |           — |            0.420 |            0.203 |            0.557 |
| ELVis         |  11,892,737 |            0.572 |            0.314 |            0.638 |
| MF-ELVis      |  47,044,608 |            0.528 |            0.279 |            0.601 |
| BRIE          |   2,940,288 |        **0.612** |        **0.348** |        **0.673** |
| BRIE-DeepSets | **209,472** | <ins>0.597</ins> | <ins>0.338</ins> | <ins>0.668</ins> |

---

### 6.4 New York

| Model         |      Params |       MRecall@10 |         MNDCG@10 |             MAUC |
| ------------- | ----------: | ---------------: | ---------------: | ---------------: |
| RND           |           — |            0.374 |            0.168 |            0.502 |
| CNT           |           — |            0.431 |            0.217 |            0.563 |
| ELVis         |  13,608,961 |            0.553 |            0.304 |            0.637 |
| MF-ELVis      |  53,909,504 |            0.516 |            0.276 |            0.602 |
| BRIE          |   3,369,344 |        **0.598** |        **0.341** |        **0.677** |
| BRIE-DeepSets | **209,472** | <ins>0.577</ins> | <ins>0.328</ins> | <ins>0.672</ins> |

---

### 6.5 Paris

| Model         |      Params |       MRecall@10 |         MNDCG@10 |             MAUC |
| ------------- | ----------: | ---------------: | ---------------: | ---------------: |
| RND           |           — |            0.459 |            0.209 |            0.502 |
| CNT           |           — |            0.499 |            0.245 |            0.557 |
| ELVis         |  12,355,329 |            0.643 |            0.352 |            0.630 |
| MF-ELVis      |  48,894,976 |            0.606 |            0.323 |            0.596 |
| BRIE          |   3,055,936 |        **0.669** |        **0.391** |        **0.666** |
| BRIE-DeepSets | **209,472** | <ins>0.661</ins> | <ins>0.375</ins> | <ins>0.661</ins> |

---

### 6.6 London

| Model         |      Params |       MRecall@10 |         MNDCG@10 |             MAUC |
| ------------- | ----------: | ---------------: | ---------------: | ---------------: |
| RND           |           — |            0.342 |            0.155 |            0.500 |
| CNT           |           — |            0.400 |            0.200 |            0.562 |
| ELVis         |  35,037,953 |            0.530 |            0.293 |            0.629 |
| MF-ELVis      | 139,625,472 |            0.531 |            0.267 |            0.597 |
| BRIE          |   8,726,592 |        **0.563** |        **0.318** |        **0.665** |
| BRIE-DeepSets | **209,472** | <ins>0.549</ins> | <ins>0.312</ins> | <ins>0.663</ins> |

---

## 7. Discussion

BRIE-DeepSets achieves the second-best ranking performance in all six evaluated cities.

Although BRIE obtains the best values across all metrics, the differences are small, especially in MAUC. This suggests that the set-based representation preserves competitive global ranking behavior while substantially reducing model size.

The gap with respect to BRIE is slightly larger for MRecall@10 and MNDCG@10 than for MAUC. This indicates that the Deep Sets representation may lose some precision in the top positions of the ranking compared with a model that stores dedicated embeddings for each user. However, this loss is limited and comes with a large scalability gain.

For example, in London, BRIE obtains a MAUC of 0.665, while BRIE-DeepSets obtains 0.663. At the same time, the number of trainable parameters is reduced from 8,726,592 to 209,472, which corresponds to a reduction of approximately **97.6%**.

The reduction is even larger when compared with ELVis and MF-ELVis, mainly because these models use larger latent dimensions in their original configurations.

An important point is that the BRIE-DeepSets configuration was selected through a limited random search on Barcelona and then reused across the remaining cities without city-specific retuning. Therefore, the reported results should not be interpreted as an exhaustive optimization for each dataset.

Overall, these results show that Deep Sets provide a competitive alternative when scalability, parameter efficiency and user cold-start capability are prioritized.

---

## 8. Main Observations

* BRIE-DeepSets removes the dependency between model size and number of users.
* The number of trainable parameters remains constant at **209,472** across all evaluated datasets.
* The original BRIE model grows linearly with the number of users.
* BRIE-DeepSets naturally supports unseen users because it does not store user-specific embeddings.
* BRIE remains the strongest method in pure ranking performance.
* BRIE-DeepSets consistently obtains the second-best results across all cities.
* The ranking gap is small, especially in MAUC.
* The scalability gain is substantial, reaching a **97.6% parameter reduction** with respect to BRIE in London.

The main contribution of this repository is therefore not a direct ranking improvement over BRIE, but a more scalable and cold-start-friendly formulation that preserves competitive recommendation performance.

---

## 9. Relationship to BRIE

This repository extends the original BRIE model and reuses its general recommendation and explainability-oriented framework.

All credit for the original BRIE architecture, training strategy and evaluation protocol belongs to the authors of the original work.

Original repository:
https://github.com/Kominaru/BRIE

---

## 10. License

This project follows the same license as the original BRIE repository.
