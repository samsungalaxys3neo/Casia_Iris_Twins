# Behind the Iris: Biometric Recognition in Twin Subjects

This repository contains the final project for the **Biometric Systems** course, developed by:

- **Sarah Oualli**
- **Gaia Diodati**
- **Michele Pogu**

Master's Degree in **Cybersecurity**  
Academic Year **2025/2026**

---

## Project Overview

This project evaluates iris recognition in twin subjects through a complete biometric pipeline. The objective is not only to build an iris recognition system, but also to study whether twin subjects represent a more challenging impostor condition than unrelated subjects.

The implemented pipeline includes:

- dataset audit and metadata extraction;
- image quality analysis;
- Daugman-style iris segmentation;
- strict segmentation filtering;
- Rubber Sheet normalization;
- conservative masking;
- Gabor-style binary feature extraction;
- masked Hamming Distance matching;
- angular shift compensation;
- verification analysis;
- statistical analysis of twin similarity;
- additional LBP and Blob/LoG texture analysis;
- closed-set identification;
- failure analysis.

The project compares genuine pairs with different impostor classes:

- unrelated subjects;
- twin same-eye pairs;
- twin cross-eye pairs;
- same-subject different-eye pairs.

---

## Repository Content

The repository contains the source code needed to reproduce the experimental pipeline.

```text
Casia_Iris_Twins/
├── scripts/
│   ├── run_pipeline.py
│   └── other pipeline scripts
├── requirements.txt
├── README.md
├── Casia_Iris_Twins_Report.pdf
└── Casia_Iris_Twins_Presentation.pdf
└── Casia_Iris_Twins_Presentation.pptx
````


The dataset and generated intermediate files are not included in the repository because they contain biometric images and large derived outputs.

---

## Dataset

The project was developed using the **CASIA-Iris-Twins** dataset.

The expected raw dataset structure is organized by family and eye folder. Each family contains four folders:

```text
family_id/
├── 1L/
├── 1R/
├── 2L/
└── 2R/
```

where:

* `1` and `2` identify the two twin subjects;
* `L` and `R` identify the left and right eye.

Example:

```text
CASIA-Iris-Twins/
├── 001/
│   ├── 1L/
│   ├── 1R/
│   ├── 2L/
│   └── 2R/
├── 002/
│   ├── 1L/
│   ├── 1R/
│   ├── 2L/
│   └── 2R/
└── ...
```

The dataset must be stored locally and passed to the pipeline using the `--raw-root` argument, put inside `data/raw/`.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/samsungalaxys3neo/Casia_Iris_Twins.git
cd Casia_Iris_Twins
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run the Full Pipeline

The full pipeline can be executed using the provided runner script:

```bash
python3 scripts/run_pipeline.py --raw-root "$(pwd)/data/raw/CASIA-Iris-Twins"
```

Example:

```bash
python scripts/run_pipeline.py --raw-root "/Users/yourname/Desktop/CASIA-Iris-Twins"
```

The script runs the main stages used in the final report:

1. dataset audit;
2. quality audit;
3. Daugman-style segmentation;
4. strict segmentation filtering;
5. Rubber Sheet normalization with conservative v3 mask;
6. Gabor-style feature extraction;
7. pair generation;
8. masked Hamming Distance matching;
9. verification analysis;
10. statistical analysis;
11. LBP and Blob/LoG texture analysis;
12. closed-set identification;
13. failure analysis.

Generated outputs are saved inside the local `data/` directory.

---

## Main Output Folders

After running the pipeline, the following folders are generated locally:

```text
data/
├── metadata/
├── quality/
├── segmentation_daugman/
├── normalized/
├── features/
├── matching/
├── analysis/
├── statistical_resampling/
├── texture_similarity/
├── identification_closed_set/
└── failure_analysis/
```

These folders contain intermediate files, plots, summaries, templates, and analysis results.

They are intentionally excluded from GitHub because they may contain biometric data or large generated files.

---

## Reproducibility Notes

The project uses a modular pipeline. Each stage produces intermediate files that are reused by the following stages. This makes the experiment easier to inspect, debug, and reproduce.

The main design choices were:

* using a strict segmentation subset instead of all automatically segmented images;
* applying a conservative mask to reduce eyelid, eyelash, reflection, and boundary artifacts;
* comparing templates only on valid bits shared by both masks;
* applying angular shift compensation during matching;
* separating impostor classes into unrelated, twin same-eye, twin cross-eye, and same-subject different-eye comparisons;
* sampling unrelated pairs to keep the experiment computationally manageable;
* adding LBP and Blob/LoG descriptors as auxiliary texture analyses.

---

## Final Report and Presentation

The final written report is available as:

```text
Casia_Iris_Twins_Report.pdf
```

The final presentation slides are available as:

```text
Casia_Iris_Twins_Presentation.pdf
```

These files summarize the methodology, design choices, experimental results, and conclusions of the project.

---

## Authors

Project developed by:

* Sarah Oualli
* Gaia Diodati
* Michele Pogu

Master's Degree in Cybersecurity
Biometric Systems
Academic Year 2025/2026

