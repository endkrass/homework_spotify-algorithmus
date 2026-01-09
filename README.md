# 🎵 Spotify Energy Prediction  
## Predictive Analytics – Group Assignment  
**Authors:** Thomas Endrass · Johannes Lehner · Lukas Mayr

---

## 🎯 Project Overview

The objective of this project was to develop a regression model to predict the energy of Spotify tracks based on audio features.  
Energy ranges from 0.0 to 1.0 and describes how fast, loud, and intense a song feels.

A provided baseline Linear Regression model achieved an R² of ~0.30.  
Our goal was to significantly outperform this baseline using a clean, reproducible, and methodologically sound machine-learning workflow, aligned with the course content.

### Final Result

- **Model:** Random Forest Regressor  
- **Parameters:**  max_depth - None, max_features - sqrt, min_samples_leaf - 1, min_samples_split - 2, n_estimators - 600
- **Performance:** **R² = 0.78**  
- **Improvement over baseline:** **~2.5× higher**

---

## 📊 Key Findings

### 1. Loudness Is the Dominant Driver of Energy

Exploratory analysis and permutation importance revealed that **loudness** is by far the most influential predictor of song energy.

- Loud tracks are almost always energetic  
- **Acousticness** shows a strong negative relationship with energy  
- Results align closely with the musical definition of energy  


![Permutation Importance](plots/correlation_with_energy.png)
📈 *See also:* `plots/permutation_importance.png`

---

### 2. Interaction Features Capture Perceptual Effects

As an initial proof of concept, we trained a simple model using only **loudness** and **acousticness**, which achieved an **R² of approximately 0.60** without hyperparameter tuning. To avoid relying exclusively on one or two dominant features, we tried to engineer additional features to capture combined perceptual effects.

The evaluated features were derived from correlation analysis during exploration and basic musical intuition:

 `loudness_tempo`  
 `danceability_valence`  
 `loudness_danceability`  
 `tempo_valence`  

- While **loudness** remained the dominant predictor, some interaction features appeared among the **Top-10**, which we saw as an encouraging result suggesting a limited but complementary contribution.

- Several early feature-engineering attempts resulted in unrealistically high R² values (up to ~0.98), which led us to identify **target leakage** sneaking repeatly in amidst the hustle. All retained interaction features were therefore engineered strictly on **X**.

| Correlation Heatmap | Permutation Importance |
|--------------------|------------------------|
| ![](plots/correlation_heatmap.png) | ![](plots/permutation_importance.png) |

---

### 3. Ensemble Models Outperform Linear Models

Six regression models were evaluated under identical conditions:

<p align="center">
<table>
<tr>
<td width="50%">

| Model | R² (Test) |
|---|---|
| **Random Forest** | **~0.78** |
| Gradient Boosting | ~0.70 |
| Linear Regression | ~0.64 |
| Ridge Regression | ~0.64 |
| ElasticNet | ~0.44 |
| Lasso | ~0.15 |
</td>
<td width="50%">
<img src="plots/model_comparison.png" width="100%">
</td>
</tr>
</table>
</p>

Tree-based ensemble models clearly outperform linear approaches, confirming **non-linear relationships** between audio features and perceived energy.

---

## 🧠 Methodology

### Data Preparation
- Removed non-informative identifier columns  
- Retained numeric features only; categorical features were tested but ultimately excluded due to stability issues within the preprocessing pipeline  
- Train/Test split in percentage: 80 / 20

### Feature Engineering & Leakage Control
- No usage of the target variable during preprocessing  
- No global statistics computed outside the training split  

### Feature Selection
- Permutation Importance used for feature ranking  
- Final model trained on the Top-10 features to balance performance and stability  

### Model Development
1. Baseline Linear Regression  
2. Systematic model comparison  
3. Feature ranking and selection  
4. Hyperparameter tuning using GridSearchCV (5-fold CV)  
5. Final training and evaluation  

---

## 🚀 Model Usage & Practical Inference

To enable a fast and convenient way to load and use the trained model, a dedicated notebook (`model_usage.ipynb`) was created, focusing on practical inference rather than model development.

The notebook is structured into three parts:

- **Model loading** – the serialized final model and the corresponding preprocessing pipeline are loaded to ensure consistent and reproducible predictions  
- **Code-based inference** – users can manually specify the required feature values in code to obtain a transparent and controllable energy prediction for a single track  
- **Interactive interface (experimental)** – an additional `ipywidgets`-based interface was implemented to explore a more user-friendly way of providing inputs and obtaining predictions  

This notebook serves as a lightweight demonstration of how the final model can be quickly reused and tested independently from the training process.



---

## 📁 Project Structure

```
homework_spotify-algorithmus/
├── data/
│ └── spotify-tracks.csv
│
├── models/
│ ├── model.pkl
│ └── top_6_indices.pkl
│
├── plots/
│ ├── ...
│
├── exploration.ipynb
├── preprocessing.ipynb
├── modelling.ipynb
├── pipeline_gridsearch.ipynb
├── train_final_model.ipynb
├── model_usage.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## 📄 File Explanation

### Folders
- **data/** – raw dataset used for model development  
- **models/** – serialized final model and selected feature indices  
- **plots/** – generated figures from EDA, feature analysis, and model evaluation  

### Notebooks
- **exploration.ipynb** – exploratory data analysis (EDA), correlation analysis, and initial feature engineering insights  
- **preprocessing.ipynb** – data cleaning, and construction of the preprocessing pipeline  
- **modelling.ipynb** – systematic comparison of multiple regression models  
- **pipeline_gridsearch.ipynb** – hyperparameter tuning and pipeline optimization using grid search (❗️runtime approx. 25 minutes due to cross-validation)

- **train_final_model.ipynb** – re-training of the final model training  
- **model_usage.ipynb** – model loading, performance overview, and interactive inference interface  


---

## 🛠️ Libraries Used

- Python 3.12  
- pandas, numpy  
- scikit-learn  
- matplotlib  
- pickle (preinstalled)  


---

## ⚠️ Disclaimer

Some minor code artifacts and unused components remain due to iterative experimentation, limited time towards the end of the project, and the fact that collaborative development via GitHub was a new learning experience for the team. The focus was therefore placed on methodological correctness, reproducibility, and a clean final model rather than cosmetic refactoring; all results and conclusions remain fully reproducible and unaffected by these artifacts.

AI-based tools were used as supportive aids for brainstorming and coding, while all modeling decisions, interpretations, and final results were developed and validated by the team.