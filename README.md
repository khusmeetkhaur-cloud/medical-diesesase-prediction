# Medical Disease Prediction

This project predicts disease category using medical and lifestyle related features.

## Dataset
The dataset contains 10000 medical records with columns like Age, Gender, BMI, BloodPressure, GlucoseLevel, Cholesterol, HeartRate, Smoking, Alcohol, PhysicalActivity, FamilyHistory and Disease.

## Models Used
- SVC
- SVR
- Decision Tree Classifier

Note: SVR is a regression model, so it is used only for comparison by rounding the predicted values.

## Project Steps
1. Import libraries
2. Load dataset
3. Check null and duplicate values
4. Basic EDA and visualization
5. Encode categorical columns
6. Split data into training and testing
7. Train SVC, SVR and Decision Tree
8. Compare model scores
9. Show confusion matrix
10. Save final model

## Result
SVC, SVR and Decision Tree were compared. Decision Tree was saved as final model because it is simple and easy to explain.

## How to Run
1. Install required libraries
2. Open the notebook
3. Run all cells

```bash
pip install -r requirements.txt
```

## Libraries
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib
