# TBM Main Bearing Early-Stage Fault Identification (ASAFormer)

This repository contains a **research codebase** for early-stage fault identification of **Tunnel Boring Machine (TBM) main bearings** from vibration signals under **weak-signal and limited-labeled-data** conditions.

The project explores a **lightweight Transformer-based** solution that aims to improve sensitivity to sparse, low-amplitude fault signatures while remaining suitable for resource-constrained monitoring.

> **Note:** This repository is prepared to support reproducibility during peer review. Some extended materials (e.g., complete ablation settings and additional implementation details) will be further organized and released after the paper is accepted.

---

## Motivation (high-level)

TBM main bearings operate at **ultra-low speeds** with **heavy combined loads** and harsh underground disturbances. During tunneling, the bearing is physically inaccessible, leading to:

* extremely weak and sparse fault-related vibration responses,
* strong background/structural noise,
* limited availability of reliably labeled fault samples.

These characteristics differ significantly from conventional bearing benchmarks and motivate measurement-aware, data-efficient modeling.

---

## What’s included

* Model implementation (ASAFormer code structure)
* Training and evaluation scripts
* Data preprocessing utilities (windowing and normalization)
* Configuration examples for small-sample settings

---

## Data availability

Due to **data sharing restrictions**, the original experimental dataset used in the paper is **not included** in this repository.
However, we provide preprocessing and loader templates to help users run the pipeline on their own vibration data. A minimal example of the expected data format (and instructions to prepare it) is provided in the code comments/configs.

If you are interested in research collaboration or academic use of the dataset, please contact the corresponding author.
