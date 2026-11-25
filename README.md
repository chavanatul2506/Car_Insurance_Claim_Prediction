# 🚗 Car Insurance Claim Prediction

Predict whether a car insurance policyholder will make a **claim**, using details about the policy, the customer and the vehicle. The project includes a full machine learning notebook and a **Streamlit** web app to try the model.

> **Problem type:** binary classification (`claim_status`: 1 = claim, 0 = no claim)

## Dataset
- **58,592 policies** and **41 columns** (customer, policy and car-specification details)
- No missing values and no duplicate rows
- **Imbalanced target:** only about **6.4%** of policies have a claim

| Group | Example columns |
|---|---|
| Policy / customer | `subscription_length`, `customer_age`, `region_code`, `region_density` |
| Vehicle | `vehicle_age`, `model`, `segment`, `fuel_type`, `engine_type`, `airbags`, `ncap_rating` |
| Safety / comfort features (Yes/No) | `is_esc`, `is_parking_sensors`, `is_speed_alert`, `is_brake_assist`, ... |
| Target | `claim_status` |

## Project workflow
1. **Data quality checks:** missing values, duplicates, class balance
2. **EDA:** distributions, claim rate by feature, Yes/No feature gap chart, categorical claim rates, correlation heatmap
3. **Feature extraction:** numbers pulled out of text columns such as `max_torque` (`250Nm@2750rpm`) and `max_power` using regular expressions
4. **Feature selection:** kept 4 numeric columns, 7 Yes/No features and the car `model`; dropped identifiers, near-duplicate car specification columns and weak features
5. **Encoding:** Yes/No converted to 1/0, car `model` one-hot encoded
6. **Train/test split:** stratified, so both parts keep the same claim rate
7. **Modelling:** Decision Tree, Random Forest and XGBoost
8. **Evaluation:** ROC-AUC, precision, recall and F1 of the claim class (accuracy is misleading when 94% of policies have no claim)
9. **Feature importance** of the best model
10. **Saving** the model with `joblib` and serving it in a Streamlit app

## Key findings
- Claims are **rare (about 6.4%)**, so accuracy alone is a poor measure. A model that always predicts "no claim" would be about 94% accurate but useless.
- **`subscription_length`** shows the strongest link to claims among the numeric columns.
- Among the Yes/No features, **`is_speed_alert`** has the largest difference in claim rate, but all differences are small (about 2 percentage points or less).
- The 11 car models fully determine every car specification column (engine, size, safety features), so those columns repeat the information in `model`.

## Results
✏️ *Fill in from your notebook after running the evaluation cell:*

| Model | ROC-AUC |
|---|---|
| Decision Tree | _ |
| Random Forest | _ |
| XGBoost | _ |

Best model: **_** &nbsp;(see the notebook for the classification report and feature importance chart)

Claims are hard to predict from this kind of data, so scores in the range of about 0.6 to 0.7 ROC-AUC are normal and should be read in that context.

## Streamlit app
The app takes the policy and vehicle details (subscription length, vehicle age, customer age, region density, car model and 7 safety features) and returns a **claim risk score** with a higher/lower risk flag.

> The score is a relative risk measure, not an exact probability, because the data is imbalanced.

![App screenshot](screenshots/app.png)
*(✏️ add a screenshot of the app in a `screenshots` folder, or delete this line)*

## Project structure
```
.
├── Car_Insurance_Claim.ipynb      # full analysis and modelling notebook
├── app.py                         # Streamlit web app
├── Insurance claims data.csv      # dataset
├── claim_model.pkl                # trained model
├── model_columns.pkl              # training column names and order
├── model_options.pkl              # list of car models for the app drop-down
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. clone the repository
git clone https://github.com/chavanatul2506/Car_Insurance_Claim_Prediction.git
cd Car_Insurance_Claim_Prediction

# 2. (optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. install the packages
pip install -r requirements.txt

# 4. start the app
streamlit run app.py
```

To re-train the model, open `Car_Insurance_Claim.ipynb` and use **Restart, then Run All**. This regenerates the three `.pkl` files.

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib, Streamlit

## Limitations and ideas for improvement
- Claim prediction from these columns is difficult, so the model is only modestly better than chance. It is a learning project and is not suitable for real pricing or underwriting decisions.
- Try hyperparameter tuning with cross-validation, new features (for example age bands), and threshold tuning for the business goal.
- The Yes/No features repeat information in `model`, so it is worth testing a model without them.

## Dataset note
Check the original source's license before reusing or redistributing the data.
