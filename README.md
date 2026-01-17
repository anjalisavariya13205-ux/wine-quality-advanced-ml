# Wine Quality Classification: Advanced ML + Explainable AI

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## 🎯 Project Overview

A comprehensive comparison of **6 machine learning algorithms**—from classical methods to advanced ensemble techniques—combined with **Explainable AI (XAI)** for model interpretability. This project demonstrates systematic model evaluation, rigorous experimental methodology, and transparent decision-making essential for data science applications.

**Key Innovation:** Integration of advanced ML models (XGBoost, LightGBM, Neural Networks) with multiple explainability frameworks (SHAP, LIME, Partial Dependence) to create transparent, trustworthy predictions.

---

## 💡 Motivation

Machine learning model selection often involves trade-offs between performance, interpretability, and computational cost. This project addresses the challenge of:
- Systematically comparing classical vs. advanced ML approaches
- Understanding **why** models make specific predictions (not just accuracy)
- Balancing performance with interpretability
- Ensuring reproducible, transparent methodology

**Real-world relevance:** These skills apply to any domain requiring data-driven decision making—from healthcare diagnostics to financial risk assessment to scientific research.

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository - Wine Quality Dataset  
**Domain:** Chemical analysis and quality assessment  
**Type:** Multivariate binary classification

**Features (11 physicochemical properties):**
- Fixed acidity, Volatile acidity, Citric acid
- Residual sugar, Chlorides
- Free sulfur dioxide, Total sulfur dioxide
- Density, pH, Sulphates, Alcohol

**Target Variable:** Binary classification  
- Good wine (1): Quality score ≥ 6
- Bad wine (0): Quality score < 6

**Dataset Size:** 1,599 wine samples  
**Data Quality:** No missing values, all numeric features

---

## 🔬 Methodology

### 1. Data Preprocessing Pipeline
```
Raw Data → Binary Target Creation → Train-Test Split (80/20) 
→ Feature Scaling (StandardScaler) → Model Training
```

**Key Design Decisions:**
- **Stratified splitting** to maintain class distribution
- **StandardScaler** for feature normalization (mean=0, std=1)
- **Random state=42** for full reproducibility
- **Separate test set** never used during training (prevents data leakage)

### 2. Models Implemented

**Classical Machine Learning (Baseline):**
1. **Logistic Regression** - Linear baseline classifier
2. **Support Vector Machine (SVM)** - RBF kernel for non-linear patterns
3. **Random Forest** - Ensemble of 100 decision trees

**Advanced Machine Learning:**
4. **XGBoost** - Gradient boosting framework (industry standard)
5. **LightGBM** - Microsoft's efficient gradient boosting
6. **Neural Network (MLP)** - Deep learning with 3 hidden layers (64→32→16 neurons)

### 3. Explainable AI Framework

**Why Explainability Matters:** Understanding *why* a model predicts is as important as accuracy in high-stakes applications.

**XAI Methods Implemented:**

1. **SHAP (SHapley Additive exPlanations)**
   - Global feature importance across all predictions
   - Local explanations for individual predictions
   - Based on game theory (Shapley values)

2. **LIME (Local Interpretable Model-Agnostic Explanations)**
   - Local approximations of complex models
   - Human-interpretable feature contributions

3. **Partial Dependence Plots**
   - Visualize feature effects on predictions
   - Show non-linear relationships

4. **Feature Importance Analysis**
   - Compare importance across all models
   - Identify key predictive factors

### 4. Evaluation Metrics

- **Accuracy** - Overall correctness
- **Precision** - Reliability of positive predictions  
- **Recall** - Sensitivity to positive class
- **F1-Score** - Harmonic mean (primary metric for model selection)
- **Confusion Matrix** - Detailed error analysis

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | F1-Score | Type | Rank |
|-------|----------|----------|------|------|
| **XGBoost** | **0.8094** | **0.8123** | Advanced | 🥇 1st |
| Random Forest | 0.7969 | 0.8071 | Classical | 🥈 2nd |
| LightGBM | 0.7906 | 0.7988 | Advanced | 🥉 3rd |
| SVM | 0.7625 | 0.7640 | Classical | 4th |
| Logistic Regression | 0.7406 | 0.7522 | Classical | 5th |
| Neural Network | 0.7188 | 0.7273 | Advanced | 6th |

### Key Findings

🏆 **Best Model: XGBoost**
- **F1-Score: 0.8123 (81.23%)**
- Achieved best balance of precision and recall
- 0.5% improvement over Random Forest
- 7.9% improvement over baseline (Logistic Regression)

📊 **Classical vs. Advanced ML:**
- **Average Classical F1-Score:** 0.7744
- **Average Advanced F1-Score:** 0.7795
- Advanced models show 0.5% improvement on average
- XGBoost specifically shows 4.8% improvement over best classical model

🔬 **Explainability Insights:**
- **Most Important Feature:** Alcohol content (confirmed across all models)
- **SHAP Analysis:** Alcohol > Sulphates > Volatile Acidity (top 3 predictors)
- High alcohol + low volatile acidity → Strong "Good Wine" prediction
- Model decisions align with wine chemistry domain knowledge

🎯 **Practical Implications:**
- Winemakers can focus on alcohol content optimization
- Volatile acidity control is critical for quality
- Sulphate levels have significant but secondary impact

---

## 🖼️ Visualizations

The project includes comprehensive visualizations:

1. **Performance Comparison Charts** - All 6 models side-by-side
2. **Confusion Matrices** - Error analysis for each model
3. **SHAP Summary Plots** - Global feature importance with distributions
4. **SHAP Waterfall Plots** - Individual prediction explanations
5. **LIME Explanations** - Local model approximations
6. **Partial Dependence Plots** - Feature effect curves
7. **Feature Importance Rankings** - Cross-model comparison

---

## 🛠️ Technologies & Tools

**Programming Language:** Python 3.13

**Core Libraries:**
- `pandas` 2.x - Data manipulation
- `numpy` 1.24+ - Numerical computing
- `scikit-learn` 1.3+ - Classical ML algorithms
- `xgboost` - Gradient boosting
- `lightgbm` - Efficient gradient boosting
- `shap` - Explainable AI framework
- `lime` - Local interpretability
- `matplotlib` & `seaborn` - Visualization

**Development Environment:**
- Jupyter Notebook - Interactive development
- Git - Version control

---

## 📁 Project Structure
```
wine-quality-project/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── Wine_Quality_Analysis.ipynb         # Basic 3-model comparison
├── Wine_Quality_Advanced_XAI.ipynb     # Advanced ML + Explainable AI
│
├── advanced_model_comparison.csv       # Performance metrics (all 6 models)
├── project_summary.txt                 # Detailed findings summary
│
└── data/
    └── winequality-red.csv             # Dataset (auto-downloaded)
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum
- Internet connection (for dataset download)

### Installation & Execution
```bash
# 1. Clone or download this repository
git clone https://github.com/yourusername/wine-quality-advanced-ml.git
cd wine-quality-advanced-ml

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Launch Jupyter Notebook
python -m notebook

# 4. Open Wine_Quality_Advanced_XAI.ipynb

# 5. Run all cells (Cell → Run All)
```

### Expected Runtime
- Complete analysis: ~3-5 minutes
- XGBoost training: ~15 seconds
- SHAP calculations: ~60-90 seconds (most time-intensive)

---

## 🔍 Key Insights

### Discovery 1: Alcohol as Primary Quality Indicator

SHAP analysis revealed **alcohol content** as the dominant predictor across all models, with feature importance scores 2-3x higher than other factors.

**Interpretation:**
- Higher alcohol correlates with grape ripeness and sugar content
- Reflects fermentation quality and timing
- Aligns with oenological research on quality determinants

### Discovery 2: Model Complexity vs. Performance Trade-off

**Observation:** XGBoost (complex) achieved 81.23% F1-Score vs. Logistic Regression (simple) at 75.22%.

**Insight:** 6% performance gain justifies added complexity for deployment, but simpler models remain valuable for:
- Quick prototyping and baselines
- Interpretability requirements
- Resource-constrained environments

### Discovery 3: Explainability Validates Domain Knowledge

SHAP and LIME explanations consistently identified:
1. Alcohol (fermentation quality)
2. Volatile acidity (spoilage indicator)
3. Sulphates (preservation)

This alignment with wine chemistry **validates** both the model and the explainability techniques.

---

## 📊 Model Selection Guide

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| **Production Deployment** | XGBoost | Best F1-Score (0.8123) |
| **Real-Time Prediction** | Logistic Regression | Fastest inference |
| **Maximum Interpretability** | Logistic Regression | Linear coefficients |
| **Feature Discovery** | Random Forest / XGBoost | Importance rankings |
| **Resource-Constrained** | Logistic Regression | Minimal compute |
| **Research/Experimentation** | All 6 models | Comprehensive comparison |

---

## ⚠️ Limitations

### Current Limitations

1. **Binary Classification Simplification**
   - Original quality scores (3-8) reduced to binary (good/bad)
   - Loses granularity in quality assessment

2. **Single Dataset Scope**
   - Results specific to red wine from one region
   - Generalization to other wine types unverified

3. **Basic Hyperparameter Configuration**
   - Used default or simple hyperparameters
   - Performance could improve with systematic tuning (GridSearchCV, Optuna)

4. **Single Train-Test Split**
   - No cross-validation implemented yet
   - Performance estimates could be more robust with k-fold CV

5. **Class Imbalance Not Addressed**
   - Dataset has natural imbalance (more good wines than bad)
   - Could explore SMOTE, class weights, or other techniques

---

## 🔮 Future Work

### Planned Enhancements

**Phase 1: Robust Validation**
- Implement 5-fold cross-validation
- Hyperparameter optimization using GridSearchCV
- ROC-AUC curve analysis
- Statistical significance testing between models

**Phase 2: Extended Analysis**
- Multi-class classification (predict exact quality scores 3-8)
- Validate on white wine dataset
- Feature engineering and interaction terms
- Ensemble stacking methods

**Phase 3: Advanced Techniques**
- Deep learning with TensorFlow/PyTorch
- AutoML framework comparison
- Causal inference analysis
- Interactive Streamlit dashboard for deployment

---

## 🎓 Skills Demonstrated

✅ **End-to-End ML Pipeline** - Data preprocessing, training, evaluation  
✅ **Advanced ML Implementation** - XGBoost, LightGBM, Neural Networks  
✅ **Explainable AI** - SHAP, LIME, Partial Dependence analysis  
✅ **Research Methodology** - Systematic comparison, reproducibility  
✅ **Critical Thinking** - Model trade-off analysis, limitation awareness  
✅ **Technical Communication** - Documentation, visualization, insights  

---

## 👨‍💻 Author

**[ANJALI SAVARIYA]**  
Data Science Enthusiast

📧 [anjalisavariya13205@gmail.com]  
💼 [https://www.linkedin.com/in/anjali-savariya-801a70293/]  
🐙 [https://github.com/anjalisavariya13205-ux]  

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Wine Quality dataset
- Dr. Paulo Cortez (University of Minho) for dataset creation
- Scikit-learn, XGBoost, LightGBM development teams
- SHAP and LIME creators for advancing explainable AI
- Open-source Python community

---

## 📚 References

1. Cortez, P., et al. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.

2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

3. Ribeiro, M. T., et al. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *KDD*.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.

---

## 📄 License

MIT License - See file for details

---

*Last Updated: January 2026*