# Avalanche Prediction Project

______________________________________________________________________________________________

### Using Data provided by weather forecasting/nowcasting and SNOWPACK ###

#Nowcast inputs: AWS measurements
#Forecast inputs: COSMO-OSHD, a numerical weather prediction (NWP) model

This project uses meteorological and snowpack data to predict wet-snow avalanche days in the Swiss Alps using multiple machine learning models and ensemble learning.

## Contents
1. `1_Plot_AvalangeOccurrence.ipynb`: Visualize avalanche days per dataset and month
2. `2_Analysis_yas.ipynb`: Exploratory data analysis (EDA) and preprocessing
3. `3_RandomForests_DecisionTree.ipynb`: Decision trees, Random Forests (F1- and recall-optimized)
4. `4_Neural_Network.ipynb`: PyTorch-based AvalancheNet
5. `5_Model_Stacking.ipynb`: Manual stacking with NN + RFs
6. `6_TemporalSplit_Model_Stacking.ipynb`: Time-aware stacking to avoid data leakage

## Data 
- `dataset1.csv`, `dataset2.csv`, `dataset3_nowcast.csv`, `dataset3_forecast.csv`: Original datasets (https://www.envidat.ch/dataset/data_wet_aval_model)

- `full_dataset.csv`: Merged, cleaned dataset used for training and evaluation

## Models
- Decision Trees and Random Forests (optimized for F1 and recall)
- PyTorch neural network (`AvalancheNet`)
- Manual model stacking with Logistic Regression and Gradient Boosting as meta-learners

## Explainability
- **SHAP**: Global and local feature importance
- **LIME**: Visual explanations for correctly and incorrectly classified days

## Highlights
- Strong performance on avalanche detection despite class imbalance
- Advanced error analysis using both SHAP and LIME
- Careful splitting to avoid data leakage and overfitting

## Known Limitations
- Random splitting may allow indirect post-avalanche patterns to influence training
- Future versions should explore seasonal or temporal splits like in `6_TemporalSplit_Model_Stacking.ipynb`
_______________________________________________________________________________________________

### Using Data provided by satellites & weather stations ###

