# Avalanche Prediction Project

Resulting Prediction Map from Satelite Data and RF : 
<img width="586" alt="Screenshot 2025-05-22 at 17 26 43" src="https://github.com/user-attachments/assets/24fda703-2261-4c22-b1f1-f79bf19a202e" />



This project builds on "Automated prediction of wet-snow avalanche activity in the Swiss Alps" (Hendrick M. et al.), which used meteorological and snowpack data with ensemble machine learning models to predict wet-snow avalanche days. We extend this approach by incorporating satellite-based snow depth estimates from Sentinel-1 and Sentinel-2. Snow depth was derived via computer vision methods developed by the ETH Remote Sensing Lab and provided by ExoLabs. Despite a limited dataset—restricted to a specific region and time period (01.04.2023 to 01.07.2023), we successfully trained a classifier capable of avalanche prediction with a spatial resolution of 10 meters and daily temporal resolution.

## Contents
1. `1_Plot_AvalangeOccurrence.ipynb`: Visualize avalanche days per dataset and month
2. `2_Exploratory_data_analaysis.ipynb`: Exploratory data analysis (EDA) and preprocessing
3. `3_logistic_regression.ipynb`: Logistic regression with/without PCA
4. `4_KNN.ipynb`: K nearest neighbors with PCA
5. `5_RandomForests_DecisionTree.ipynb`: Decision trees, Random Forests (F1- and recall-optimized)
6. `6_Neural_Network.ipynb`: PyTorch-based AvalancheNet
7. `7_Model_Stacking.ipynb`: Manual stacking with NN + RFs
8. `8_TemporalSplit_Model_Stacking.ipynb`: Time-aware stacking to avoid data leakage
9. `9_Prepare_sat_ds.ipynb`: Getting data from all sources and processing into one dataset
10. `10_Predict_avalanche_sat.ipynb`: Training RF and generating "Prediction Map" with it 

## Data 
- `dataset1.csv`, `dataset2.csv`, `dataset3_nowcast.csv`, `dataset3_forecast.csv`: Original datasets (https://www.envidat.ch/dataset/data_wet_aval_model)

- `full_dataset.csv`: Merged, cleaned dataset used for training and evaluation
- '/Data/Satelite-Data/data_clip_exolabs': Everyday snapshots of certain alpine region with snow-depth estiamtions over 3 month (availible on request)
- '/Data/Satelite-Data/meteoswiss_data' : data exported from 12 meteo-stations located within the bounds of before-mentioned 
- '/Data/Satelite-Data/dem.tif' : digital elevation map

## Models
- KNN, Logistic regression with PCA/without
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
- Careful about multicollinearity
- Proposing approaches for Explainable AI and Active learning

## Known Limitations
- Random splitting may allow indirect post-avalanche patterns to influence training
- Future versions should explore seasonal or temporal splits like in `6_TemporalSplit_Model_Stacking.ipynb`
_______________________________________________________________________________________________

### Using Data provided by satellites & weather stations ###

