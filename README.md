<div align="center">

# ⚗️ Distillation Column — Ethanol Purity Predictor

### Real-Time Machine Learning Estimation of Ethanol Purity in Distillation Columns

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-189FDD?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/Mausam5055/Distillation-Column-Prediction?style=social)](https://github.com/Mausam5055/Distillation-Column-Prediction/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Mausam5055/Distillation-Column-Prediction?style=social)](https://github.com/Mausam5055/Distillation-Column-Prediction/network/members)
[![GitHub issues](https://img.shields.io/github/issues/Mausam5055/Distillation-Column-Prediction?style=social)](https://github.com/Mausam5055/Distillation-Column-Prediction/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/Mausam5055/Distillation-Column-Prediction?style=flat-square&color=blue)](https://github.com/Mausam5055/Distillation-Column-Prediction/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/Mausam5055/Distillation-Column-Prediction?style=flat-square&color=purple)](https://github.com/Mausam5055/Distillation-Column-Prediction)

<br/>

> **A physics-informed machine learning system** that transforms 7 raw operating parameters into 21 engineered features — predicting ethanol purity with **R² > 0.98** accuracy.

<br/>

<img src="feature_importance.png" alt="Feature Importance Chart" width="700"/>

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [ML Pipeline Flowchart](#-ml-pipeline-flowchart)
- [Application User Flow](#-application-user-flow)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Input Parameters](#-input-parameters)
- [Output Interpretation](#-output-interpretation)
- [Usage Tips](#-usage-tips)
- [Model Training](#-model-training-colab)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This application predicts **ethanol purity (mole fraction)** in distillation columns using ensemble machine learning models. It was trained on **4,000+ real distillation experiments** and validated with 5-fold cross-validation across diverse operating conditions.

The system accepts 7 readily-available operating parameters from DCS/SCADA systems and auto-calculates 14 additional physics-based features — delivering an instant purity estimate that engineers can trust for operational guidance.

| Aspect | Detail |
|:---|:---|
| **Domain** | Chemical Engineering — Distillation & Separation |
| **Task** | Regression — Predict Ethanol Mole Fraction |
| **Input** | 7 Core Operating Parameters |
| **Engineered Features** | 21 Total (7 raw + 14 derived) |
| **Models** | Random Forest & XGBoost (best model auto-selected) |
| **Accuracy** | R² > 0.98 · RMSE 0.0155 · MAE 0.0122 |
| **Interface** | Interactive Streamlit Web Dashboard |

---

## ✨ Key Features

| Feature | Description |
|:---|:---|
| ⚡ **Real-Time Prediction** | Instant ethanol purity estimation from live operating data |
| 🧪 **Physics-Informed** | Auto-calculates engineering ratios (Reflux Ratio, Reboiler Intensity) and temperature approximations |
| 📊 **Single Dashboard** | Consolidated metrics, predictions, and feature analysis in one view |
| 🎯 **High Accuracy** | R² > 0.98 with ±1.55% error margin |
| 🔍 **Explainable AI** | Feature Importance visualization shows *why* the model made each prediction |
| 🏭 **Industry-Ready** | Accepts parameters directly from DCS/SCADA systems |
| 🎨 **Color-Coded Status** | Instantly identifies OPTIMAL / ACCEPTABLE / LOW purity conditions |

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph INPUT["📥 User Input Layer"]
        A["Pressure P"]
        B["Temperature T1"]
        C["Reflux Flow L"]
        D["Vapor Flow V"]
        E["Distillate D"]
        F["Bottoms B"]
        G["Feed Flow F"]
    end

    subgraph ENGINE["⚙️ Feature Engineering Engine"]
        direction TB
        H["Derived Ratios: Reflux, Reboiler, Condenser, Feed Norm, D/F, B/F"]
        I["Interaction Terms: Reflux x Temp, Reboiler x Temp, Feed x Reflux"]
        J["Efficiency: Column Load, Separation Duty, Column Efficiency"]
        K["Temp Calc: T_bottom = T1 + 20C"]
    end

    subgraph MODEL["🧠 ML Prediction Layer"]
        L["StandardScaler — Normalize 21 Features"]
        M["Random Forest / XGBoost — Best Auto-Selected"]
    end

    subgraph OUTPUT["📤 Output Layer"]
        N["OPTIMAL: Purity >= 82%"]
        O["ACCEPTABLE: Purity 75-82%"]
        P["LOW PURITY: Below 75%"]
    end

    INPUT --> ENGINE
    ENGINE --> MODEL
    MODEL --> OUTPUT

    style INPUT fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    style ENGINE fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    style MODEL fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    style OUTPUT fill:#FCE4EC,stroke:#C62828,stroke-width:2px,color:#B71C1C
    style A fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style B fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style C fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style D fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style E fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style F fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style G fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style H fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style I fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style J fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style K fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style L fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    style M fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    style N fill:#FFCDD2,stroke:#C62828,color:#B71C1C
    style O fill:#FFCDD2,stroke:#C62828,color:#B71C1C
    style P fill:#FFCDD2,stroke:#C62828,color:#B71C1C
```

---

## 🔄 ML Pipeline Flowchart

```mermaid
flowchart TD
    subgraph PHASE1["Phase 1: Data Preparation"]
        A["Load Raw Dataset — 4000+ experiments"] --> B["Clean Data — Remove duplicates and nulls"]
        B --> C["Remove Outliers — Keep 1st to 99th percentile"]
        C --> D["Engineer 21 Features — Ratios, interactions, efficiency"]
        D --> E["Balance Dataset — Undersample middle range to 40%"]
    end

    subgraph PHASE2["Phase 2: Model Development"]
        F["Split Data — Train 60% Val 20% Test 20%"] --> G["Scale Features — StandardScaler on train only"]
        G --> H["Train Models — XGBoost + Random Forest"]
        H --> I["5-Fold Cross-Validation — Evaluate on training set"]
        I --> J["Select Best Model — Compare test R2 scores"]
    end

    subgraph PHASE3["Phase 3: Deployment"]
        K["Export Artifacts — model.pkl, scaler.pkl, features_names.pkl"] --> L["Streamlit App — Real-time prediction dashboard"]
    end

    PHASE1 --> PHASE2
    PHASE2 --> PHASE3

    style PHASE1 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    style PHASE2 fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    style PHASE3 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    style A fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style B fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style C fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style D fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style E fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    style F fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style G fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style H fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style I fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style J fill:#FFE0B2,stroke:#E65100,color:#BF360C
    style K fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    style L fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
```

---

## 🧑‍💻 Application User Flow

```mermaid
flowchart TD
    START(["User Opens App at localhost:8501"]) --> LOAD{"Model Files Found?"}

    LOAD -- "No" --> ERR["Display Error: Missing pkl files"]
    ERR --> STOP(["App Stops"])

    LOAD -- "Yes" --> INPUT["Enter 7 Parameters: P, T1, L, V, D, B, F"]
    INPUT --> CALC["Auto-Calculate 14 Derived Features"]
    CALC --> METRICS["Display Live Metrics: Reflux, Reboiler, Condenser"]
    METRICS --> BTN{"User Clicks Run Simulation?"}

    BTN -- "Not yet" --> INPUT

    BTN -- "Yes" --> SCALE["Scale 21 Features with StandardScaler"]
    SCALE --> PREDICT["Model Prediction: Random Forest inference"]
    PREDICT --> CHECK{"Purity Value?"}

    CHECK -- "Above 82%" --> GREEN["OPTIMAL — Maintain current settings"]
    CHECK -- "75% to 82%" --> YELLOW["ACCEPTABLE — Monitor and adjust"]
    CHECK -- "Below 75%" --> RED["LOW PURITY — Check reflux, temp, feed"]

    GREEN --> DISPLAY["Show Result Card: Purity + Status + Color"]
    YELLOW --> DISPLAY
    RED --> DISPLAY
    DISPLAY --> INPUT

    style START fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    style STOP fill:#FFCDD2,stroke:#C62828,stroke-width:2px,color:#B71C1C
    style ERR fill:#FFCDD2,stroke:#C62828,stroke-width:2px,color:#B71C1C
    style LOAD fill:#FFF9C4,stroke:#F57F17,stroke-width:2px,color:#E65100
    style INPUT fill:#BBDEFB,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    style CALC fill:#FFE0B2,stroke:#E65100,stroke-width:2px,color:#BF360C
    style METRICS fill:#E1BEE7,stroke:#7B1FA2,stroke-width:2px,color:#4A148C
    style BTN fill:#FFF9C4,stroke:#F57F17,stroke-width:2px,color:#E65100
    style SCALE fill:#B2EBF2,stroke:#00838F,stroke-width:2px,color:#006064
    style PREDICT fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    style CHECK fill:#FFF9C4,stroke:#F57F17,stroke-width:2px,color:#E65100
    style GREEN fill:#C8E6C9,stroke:#1B5E20,stroke-width:2px,color:#1B5E20
    style YELLOW fill:#FFF9C4,stroke:#F57F17,stroke-width:2px,color:#E65100
    style RED fill:#FFCDD2,stroke:#B71C1C,stroke-width:2px,color:#B71C1C
    style DISPLAY fill:#B3E5FC,stroke:#0277BD,stroke-width:2px,color:#01579B
```

---

## 🧬 Feature Engineering

The core innovation of this project lies in its **physics-informed feature engineering**. Instead of feeding raw sensor readings directly, the system constructs 21 meaningful features that capture the thermodynamics and mass-transfer physics of a distillation column.

### Feature Categories

| # | Feature Name | Category | Formula / Description |
|:---:|:---|:---:|:---|
| 1 | `Pressure` | 🔵 Core | Operating pressure (bar) |
| 2 | `L` | 🔵 Core | Reflux flow rate (kmol/hr) |
| 3 | `V` | 🔵 Core | Vapor flow rate (kmol/hr) |
| 4 | `D` | 🔵 Core | Distillate flow rate (kmol/hr) |
| 5 | `B` | 🔵 Core | Bottoms flow rate (kmol/hr) |
| 6 | `F` | 🔵 Core | Feed flow rate (kmol/hr) |
| 7 | `Temp_Bottom` | 🟠 Reference | T₁ + 20°C (reboiler approximation) |
| 8 | `Reflux_Ratio` | 🟢 Derived | L / V |
| 9 | `Reboiler_Intensity` | 🟢 Derived | V / F |
| 10 | `Condenser_Load` | 🟢 Derived | L / F |
| 11 | `Feed_Normalized` | 🟢 Derived | F / Mean(F_train) |
| 12 | `Distillate_Withdrawal` | 🟢 Derived | D / F |
| 13 | `Bottoms_Withdrawal` | 🟢 Derived | B / F |
| 14 | `Column_Load` | 🟣 Efficiency | (L + V) / F |
| 15 | `Reflux_x_Temp_Top` | 🔴 Interaction | Reflux_Ratio × T₁ |
| 16 | `Reflux_x_Temp_Diff` | 🔴 Interaction | Reflux_Ratio × (T_bottom − T_top) |
| 17 | `Reboiler_x_Temp_Bottom` | 🔴 Interaction | Reboiler_Intensity × T_bottom |
| 18 | `Feed_x_Reflux` | 🔴 Interaction | Feed_Normalized × Reflux_Ratio |
| 19 | `Feed_x_Reboiler` | 🔴 Interaction | Feed_Normalized × Reboiler_Intensity |
| 20 | `Separation_Duty` | 🟣 Efficiency | Reflux_Ratio × Reboiler_Intensity |
| 21 | `Column_Efficiency` | 🟣 Efficiency | Reflux_Ratio × Column_Load |

### Feature Category Summary

| Category | Count | Purpose |
|:---|:---:|:---|
| 🔵 Core Parameters | 6 | Direct sensor readings from DCS/SCADA |
| 🟠 Temperature Reference | 1 | Approximated reboiler temperature |
| 🟢 Derived Ratios | 6 | Normalized operating conditions |
| 🔴 Interaction Terms | 5 | Capture thermodynamic coupling effects |
| 🟣 Efficiency Metrics | 3 | Combined column performance indicators |

---

## 📈 Model Performance

### Model Comparison

| Metric | Random Forest | XGBoost | Winner |
|:---|:---:|:---:|:---:|
| **R² Score** | ~0.98 | ~0.97 | 🏆 Random Forest |
| **RMSE** | 0.0155 | 0.0178 | 🏆 Random Forest |
| **MAE** | 0.0122 | 0.0141 | 🏆 Random Forest |
| **5-Fold CV (Mean R²)** | ~0.97 | ~0.96 | 🏆 Random Forest |

### Model Specifications

| Property | Value |
|:---|:---|
| **Selected Model** | Random Forest Regressor |
| **Estimators** | 200 trees |
| **Max Depth** | 20 |
| **Min Samples Split** | 5 |
| **Training Samples** | ~1,200 (balanced) |
| **Original Dataset** | 4,000+ experiments |
| **Cross-Validation** | 5-Fold (shuffled) |
| **Scaling** | StandardScaler (fit on train only) |

### Validation Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                        Full Dataset (4,000+)                     │
├────────────────────┬──────────────────────────────────────────────┤
│  Cleaned & Balanced │  ~1,200 rows after outlier removal &       │
│                     │  undersampling of dominant middle range     │
├──────────┬─────────┴──────────┬───────────────────────────────────┤
│  Train   │    Validation      │            Test                   │
│  60%     │       20%          │            20%                    │
│          │                    │                                   │
│ Scaler   │  Hyperparameter    │   Final unbiased                  │
│ fitted   │  tuning & model    │   performance                     │
│ here     │  selection         │   evaluation                      │
└──────────┴────────────────────┴───────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|:---|:---|
| Python | 3.8 or higher |
| pip | Latest recommended |
| Git | Any recent version |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mausam5055/Distillation-Column-Prediction.git
cd Distillation-Column-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```

The app will open automatically at **`http://localhost:8501`**

---

## 📁 Project Structure

```
Distillation-Column-Prediction/
│
├── 📄 README.md                    # Project documentation (this file)
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitattributes               # Git LFS / attribute config
│
├── 🐍 app.py                      # Main Streamlit application (509 lines)
├── 🐍 model_training_script.py    # Full training pipeline (Colab-ready)
│
├── 🤖 model.pkl                   # Trained Random Forest model (~8 MB)
├── 📏 scaler.pkl                  # Fitted StandardScaler
├── 📋 features_names.pkl          # Ordered list of 21 feature names
├── 📊 feature_importance.png      # Feature importance bar chart
│
├── 📂 sample_data/
│   └── 📄 dataset_distill.csv     # Training dataset (~579 KB)
│
└── 📂 .devcontainer/
    └── 📄 devcontainer.json       # GitHub Codespaces configuration
```

### File Descriptions

| File | Size | Purpose |
|:---|:---:|:---|
| `app.py` | 17 KB | Streamlit web app — input UI, feature engineering, prediction display |
| `model_training_script.py` | 13 KB | End-to-end ML pipeline: cleaning → engineering → training → export |
| `model.pkl` | ~8 MB | Serialized best model (Random Forest with 200 estimators) |
| `scaler.pkl` | 1.3 KB | StandardScaler fitted on training data's 21 features |
| `features_names.pkl` | 329 B | Ordered list ensuring inference matches training feature order |
| `dataset_distill.csv` | 579 KB | Semicolon-delimited dataset with temperatures in Kelvin |

---

## 📥 Input Parameters

The app accepts **7 operating parameters** commonly available from process control systems:

| # | Parameter | Symbol | Unit | Range | Description |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Pressure** | P | bar | 0.5 – 3.0 | Column operating pressure |
| 2 | **Top Temperature** | T₁ | °C | 60 – 120 | Temperature at the top tray sensor |
| 3 | **Reflux Flow** | L | kmol/hr | 300 – 1,200 | Liquid returned from the condenser |
| 4 | **Vapor Flow** | V | kmol/hr | 600 – 1,500 | Vapor rising from the reboiler |
| 5 | **Distillate Flow** | D | kmol/hr | 100 – 500 | Top product withdrawal rate |
| 6 | **Bottoms Flow** | B | kmol/hr | 100 – 500 | Bottom product withdrawal rate |
| 7 | **Feed Flow** | F | kmol/hr | 350 – 700 | Raw material feed rate |

### Typical Operating Ranges

| Parameter | Normal | ⚠️ Critical High | ⚠️ Critical Low |
|:---|:---:|:---:|:---:|
| Top Temp (T₁) | 77 – 80 °C | > 85 °C *(water in product)* | < 75 °C *(subcooled)* |
| Reflux Ratio (L/V) | 1.2 – 2.5 | > 3.0 *(flooding risk)* | < 0.6 *(poor separation)* |
| Reboiler Vapor (V) | 900 – 1,100 | > 1,200 | < 800 |
| Mass Balance (D+B) | ≈ F | D+B >> F | D+B << F |

---

## 📤 Output Interpretation

### Purity Status Levels

| Status | Purity Range | Color | Recommended Action |
|:---|:---:|:---:|:---|
| 🟢 **OPTIMAL** | ≥ 82% | Green | Maintain current settings |
| 🟡 **ACCEPTABLE** | 75% – 82% | Yellow | Monitor — consider increasing reflux |
| 🔴 **LOW PURITY** | < 75% | Red | Immediate attention — check reflux, temperature, and feed rate |

### Typical Prediction Ranges

| Purity Range | Operating Condition | Likely Cause |
|:---|:---|:---|
| 0.92 – 0.98 | Excellent separation | High reflux ratio, optimal temperatures |
| 0.85 – 0.92 | Normal operation | Standard reflux, good conditions |
| 0.75 – 0.85 | Acceptable | Standard operation, moderate reflux |
| < 0.75 | Poor separation | Low reflux, high top temperature, column overload |

---

## 💡 Usage Tips

### Best Practices

| Tip | Explanation |
|:---|:---|
| ✅ Use **steady-state** data | Transient readings during startup/shutdown will give unreliable predictions |
| ✅ Maintain **mass balance** | Ensure D + B ≈ F — large imbalances indicate sensor errors |
| ✅ Stay within **operating ranges** | Extrapolation beyond training ranges decreases reliability |
| ✅ Keep **T₁ sensor calibrated** | Top temperature is a critical driver of prediction accuracy |
| ✅ Prefer **higher reflux ratios** | Higher reflux generally yields higher predicted purity |

### Troubleshooting

| Problem | Possible Cause | Fix |
|:---|:---|:---|
| Purity < 0.70 | Reflux Ratio too low | Increase reflux flow (L) |
| Purity < 0.70 | Top Temp > 82°C | Reduce reboiler steam (V) — water is boiling into the product |
| Purity < 0.70 | Feed > 620 kmol/hr | Column may be overloaded — reduce feed rate |
| Model file error | Missing `.pkl` files | Ensure `model.pkl`, `scaler.pkl`, `features_names.pkl` are in the root directory |
| Inconsistent results | Sensor drift | Recalibrate T₁ temperature sensor |

---

## 🎓 Model Training (Colab)

The training pipeline is contained in `model_training_script.py` and is designed to run on **Google Colab**.

### Training Steps

| Step | Description | Key Details |
|:---:|:---|:---|
| 1 | **Load Data** | Read `dataset_distill.csv` (semicolon-delimited) |
| 2 | **Unit Conversion** | Convert temperatures T1–T14 from Kelvin → Celsius |
| 3 | **Data Cleaning** | Remove duplicates, nulls, and outliers (1st–99th percentile) |
| 4 | **Feature Engineering** | Create 21 features from 6 core parameters + temperature |
| 5 | **Dataset Balancing** | Undersample dominant middle purity range (40%) |
| 6 | **Train/Val/Test Split** | 60% train · 20% validation · 20% test |
| 7 | **Scaling** | StandardScaler fit on training data only |
| 8 | **Model Training** | Train XGBoost (500 trees) and Random Forest (200 trees) |
| 9 | **Cross-Validation** | 5-Fold CV on training set for both models |
| 10 | **Model Selection** | Auto-select the model with the highest test R² |
| 11 | **Residual Analysis** | Predicted vs Actual, Q-Q Plot, Homoscedasticity check |
| 12 | **Export Artifacts** | Save `model.pkl`, `scaler.pkl`, `features_names.pkl` |

### Model Hyperparameters

<details>
<summary><b>🌲 Random Forest Configuration</b></summary>

| Parameter | Value |
|:---|:---|
| `n_estimators` | 200 |
| `max_depth` | 20 |
| `min_samples_split` | 5 |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |

</details>

<details>
<summary><b>⚡ XGBoost Configuration</b></summary>

| Parameter | Value |
|:---|:---|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.03 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `random_state` | 42 |

</details>

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|:---|:---|:---|
| **Language** | Python 3.8+ | Core programming language |
| **Web Framework** | Streamlit | Interactive dashboard UI |
| **ML Libraries** | scikit-learn, XGBoost | Model training & inference |
| **Data Processing** | pandas, NumPy | Data manipulation & computation |
| **Visualization** | Matplotlib, Seaborn | Feature importance & residual plots |
| **Image Processing** | Pillow (PIL) | Display feature importance chart |
| **Serialization** | pickle | Model & scaler persistence |
| **Development** | Google Colab | Training environment |
| **Deployment** | GitHub Codespaces | Cloud development via devcontainer |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💾 Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **📤 Push** to the branch (`git push origin feature/amazing-feature`)
5. **🔀 Open** a Pull Request

### Contribution Ideas

| Area | Suggestion |
|:---|:---|
| 🧪 Data | Add more distillation experiments or real plant data |
| 🤖 Models | Try neural networks, gradient boosting variants, or ensemble stacking |
| 🎨 UI | Add trend charts, historical predictions, or batch upload |
| 📊 Analysis | Add SHAP values for per-prediction explainability |
| 🚀 Deployment | Add Docker support or cloud deployment (AWS/GCP/Azure) |

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License · Copyright (c) 2026 Krishna Narayan Singh
```

---

## 📞 Contact

| Channel | Link |
|:---|:---|
| 📧 **Email** | [krishnanarayansingh65@gmail.com](mailto:krishnanarayansingh65@gmail.com) |
| 💼 **LinkedIn** | [Krishna Narayan Singh](https://www.linkedin.com/in/krishnansingh) |
| 🐙 **GitHub (Author)** | [KrishnaNsingh](https://github.com/KrishnaNsingh) |
| 🐙 **GitHub (Contributor)** | [Mausam5055](https://github.com/Mausam5055) |

---

<div align="center">

**Built with ❤️ for Chemical Engineering & Machine Learning**

**Author:** Krishna Narayan Singh · **Last Updated:** April 2026 · **Version:** 1.0

[![Star this repo](https://img.shields.io/badge/⭐_Star_this_repo-If_it_helped_you!-yellow?style=for-the-badge)](https://github.com/Mausam5055/Distillation-Column-Prediction)

</div>
