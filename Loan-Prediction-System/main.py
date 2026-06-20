import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
)

# Define schema and features
NUMERIC_FEATURES = [
    'annual_inc', 
    'dti', 
    'revol_bal', 
    'revol_util', 
    'open_acc', 
    'total_acc', 
    'pub_rec'
]
CATEGORICAL_FEATURES = [
    'home_ownership', 
    'purpose', 
    'term', 
    'verification_status', 
    'application_type'
]

# All columns required for either regression or classification task
ALL_COLS = ['loan_amnt', 'loan_status'] + NUMERIC_FEATURES + CATEGORICAL_FEATURES

def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Cleans the input DataFrame, handling data types and stripping whitespace.
    """
    df = df.copy()
    
    # Strip whitespace from categorical string columns if present
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Ensure revol_util is numeric (convert strings with % or coerce errors)
    if 'revol_util' in df.columns:
        if df['revol_util'].dtype == object:
            df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '', regex=False)
        df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')
        
    # Coerce numeric features
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Coerce loan_amnt
    if 'loan_amnt' in df.columns:
        df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
        
    return df

def build_loan_prediction_system(dataset_path: str, task: str = 'regression', sample_size: int = None):
    """
    Loads dataset, preprocesses, trains Linear/Logistic Regression, and prints evaluation metrics.
    """
    print(f"[*] Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"[!] Error: File '{dataset_path}' does not exist.")
        sys.exit(1)
        
    # Read only required columns to optimize memory
    try:
        if sample_size and sample_size > 0:
            print(f"[*] Reading a random sample of {sample_size:,} rows to optimize training time...")
            full_df = pd.read_csv(dataset_path, usecols=ALL_COLS)
            df = full_df.sample(n=min(sample_size, len(full_df)), random_state=42)
            del full_df # free memory
        else:
            print("[*] Reading full dataset...")
            df = pd.read_csv(dataset_path, usecols=ALL_COLS)
    except Exception as e:
        print(f"[!] Error reading dataset: {e}")
        sys.exit(1)

    print(f"[*] Loaded {len(df):,} rows. Cleaning data...")
    df = clean_data(df, is_training=True)
    
    if task == 'classification':
        print("[*] Processing target for classification (binary mapping of loan_status)...")
        # Define binary target mapping
        status_map = {
            'Fully Paid': 1,
            'Does not meet the credit policy. Status:Fully Paid': 1,
            'Charged Off': 0,
            'Default': 0,
            'Does not meet the credit policy. Status:Charged Off': 0
        }
        # Filter to completed loan statuses
        df = df[df['loan_status'].isin(status_map.keys())].copy()
        df['target'] = df['loan_status'].map(status_map)
        
        # loan_amnt becomes a feature for classification
        num_features = NUMERIC_FEATURES + ['loan_amnt']
        cat_features = CATEGORICAL_FEATURES
        target_col = 'target'
        
        # Drop rows where target is missing
        df = df.dropna(subset=[target_col])
        print(f"[*] Filtered and mapped data. Row count after classification mapping: {len(df):,}")
    else:
        # Regression task
        df = df.dropna(subset=['loan_amnt'])
        num_features = NUMERIC_FEATURES
        cat_features = CATEGORICAL_FEATURES
        target_col = 'loan_amnt'
        print(f"[*] Row count after cleaning: {len(df):,}")
        
    X = df[num_features + cat_features]
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[*] Split data: Training set size = {len(X_train):,}, Testing set size = {len(X_test):,}")
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    # Final Model Pipeline
    if task == 'classification':
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        print("[*] Training Logistic Regression classifier...")
    else:
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        print("[*] Training Linear Regression model...")
        
    model_pipeline.fit(X_train, y_train)
    print("[*] Model trained. Evaluating model...")
    
    predictions = model_pipeline.predict(X_test)
    
    metrics = {}
    if task == 'classification':
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}
        
        print("\n" + "="*40)
        print("         EVALUATION METRICS (CLASSIFICATION)")
        print("="*40)
        print(f"Accuracy                   : {accuracy:.4%}")
        print(f"Precision (Class 1 - Paid) : {precision:.4%}")
        print(f"Recall (Class 1 - Paid)    : {recall:.4%}")
        print("\nConfusion Matrix:")
        print(f"               Predicted Default    Predicted Paid")
        print(f"Actual Default   {cm[0][0]:<19}  {cm[0][1]:<19}")
        print(f"Actual Paid      {cm[1][0]:<19}  {cm[1][1]:<19}")
        print("\nClassification Report:")
        print(report)
        print("="*40)
    else:
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        print("\n" + "="*40)
        print("         EVALUATION METRICS (REGRESSION)")
        print("="*40)
        print(f"Mean Absolute Error (MAE)  : ${mae:,.2f}")
        print(f"Root Mean Squared Error    : ${rmse:,.2f}")
        print(f"R-squared (R2 Score)       : {r2:.4f}")
        print("="*40)
        
    # Attach task type to pipeline object so it persists
    model_pipeline.task_type = task
    
    # Analyze and print coefficients
    try:
        # Get feature names from ColumnTransformer
        feature_names = preprocessor.get_feature_names_out()
        if task == 'classification':
            coefficients = model_pipeline.named_steps['classifier'].coef_[0]
            intercept = model_pipeline.named_steps['classifier'].intercept_[0]
        else:
            coefficients = model_pipeline.named_steps['regressor'].coef_
            intercept = model_pipeline.named_steps['regressor'].intercept_
            
        # Strip prefixes (like 'num__' or 'cat__') for readability
        clean_feature_names = []
        for name in feature_names:
            if name.startswith('num__'):
                clean_feature_names.append(name[5:])
            elif name.startswith('cat__'):
                clean_feature_names.append(name[5:])
            else:
                clean_feature_names.append(name)
                
        coef_df = pd.DataFrame({
            'Feature': clean_feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values(by='Abs_Coefficient', ascending=False)
        
        print("\n" + "="*45)
        if task == 'classification':
            print(f"  TOP LOGISTIC REGRESSION COEFFICIENTS (Intercept: {intercept:.4f})")
            print("  (Positive: increases probability of repayment/Fully Paid)")
            print("  (Negative: increases probability of default/Charged Off)")
            print("="*45)
            for _, row in coef_df.head(15).iterrows():
                sign = "+" if row['Coefficient'] >= 0 else "-"
                print(f"{row['Feature']:<35} : {sign} {abs(row['Coefficient']):.4f}")
        else:
            print(f"  TOP LINEAR REGRESSION COEFFICIENTS (Intercept: ${intercept:,.2f})")
            print("="*45)
            for _, row in coef_df.head(15).iterrows():
                sign = "+" if row['Coefficient'] >= 0 else "-"
                print(f"{row['Feature']:<35} : {sign} ${abs(row['Coefficient']):,.2f}")
        print("="*45)
    except Exception as e:
        print(f"[!] Warning: Could not analyze coefficients: {e}")
        
    return model_pipeline, metrics

def load_model_safe(model_path: str):
    """
    Loads a joblib model and patches any missing attributes (like multi_class in LogisticRegression)
    to prevent version incompatibility errors between different scikit-learn versions.
    """
    model = joblib.load(model_path)
    
    # Try to patch LogisticRegression's deprecated/removed multi_class attribute if it's missing
    try:
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            clf = model.named_steps['classifier']
            if clf and not hasattr(clf, 'multi_class'):
                clf.multi_class = 'auto'
    except Exception:
        pass
        
    return model

def predict_single(model_pipeline, input_dict: dict, task: str = 'regression') -> dict:
    """
    Predicts for a single input dictionary.
    """
    df = pd.DataFrame([input_dict])
    df = clean_data(df, is_training=False)
    
    # Re-order columns to match features
    if task == 'classification':
        features = NUMERIC_FEATURES + ['loan_amnt'] + CATEGORICAL_FEATURES
    else:
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        
    df = df[features]
    
    if task == 'classification':
        pred = model_pipeline.predict(df)[0]
        probs = model_pipeline.predict_proba(df)[0]
        return {
            'prediction': int(pred),
            'probability_default': float(probs[0]),
            'probability_paid': float(probs[1])
        }
    else:
        pred = model_pipeline.predict(df)[0]
        return {
            'prediction': float(pred)
        }

def interactive_prediction(model_pipeline):
    """
    Runs an interactive CLI prompt to collect inputs and run prediction.
    """
    task = getattr(model_pipeline, 'task_type', 'regression')
    print("\n" + "="*50)
    print(f"   INTERACTIVE LOAN {'STATUS PREDICTION' if task == 'classification' else 'AMOUNT PREDICTION'}")
    print("="*50)
    print("Please enter the following details:")
    
    inputs = {}
    
    # Collect Numeric inputs
    numeric_prompts = {}
    if task == 'classification':
        numeric_prompts['loan_amnt'] = "Requested Loan Amount (e.g., 15000): "
        
    numeric_prompts.update({
        'annual_inc': "Annual Income (e.g., 65000): ",
        'dti': "Debt-to-Income Ratio (e.g., 18.5): ",
        'revol_bal': "Revolving Credit Balance (e.g., 12000): ",
        'revol_util': "Revolving Line Utilization % (e.g., 45.2): ",
        'open_acc': "Number of Open Credit Lines (e.g., 10): ",
        'total_acc': "Total Number of Credit Lines (e.g., 25): ",
        'pub_rec': "Number of Derogatory Public Records (e.g., 0): "
    })
    
    for feature, prompt in numeric_prompts.items():
        while True:
            val_str = input(prompt).strip()
            if not val_str:
                inputs[feature] = 0.0
                break
            try:
                inputs[feature] = float(val_str)
                break
            except ValueError:
                print("[!] Invalid number. Please try again.")
                
    # Collect Categorical inputs
    categorical_choices = {
        'home_ownership': ("Home Ownership", ['MORTGAGE', 'RENT', 'OWN', 'ANY', 'OTHER']),
        'purpose': ("Loan Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical', 'car', 'small_business']),
        'term': ("Loan Term", ["36 months", "60 months"]),
        'verification_status': ("Verification Status", ["Not Verified", "Source Verified", "Verified"]),
        'application_type': ("Application Type", ["Individual", "Joint App"])
    }
    
    for feature, (label, choices) in categorical_choices.items():
        print(f"\nSelect {label}:")
        for idx, choice in enumerate(choices, 1):
            print(f"  {idx}) {choice}")
        while True:
            choice_str = input(f"Enter choice (1-{len(choices)}): ").strip()
            try:
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(choices):
                    inputs[feature] = choices[choice_idx]
                    break
                else:
                    print(f"[!] Please enter a number between 1 and {len(choices)}.")
            except ValueError:
                print("[!] Invalid option. Please enter a number.")
                
    pred_res = predict_single(model_pipeline, inputs, task=task)
    print("\n" + "="*50)
    if task == 'classification':
        status_label = "Fully Paid (Repayment likely)" if pred_res['prediction'] == 1 else "Charged Off / Default (High Risk)"
        print(f" >>> Predicted Loan Status: {status_label} <<<")
        print(f" >>> Probability of repayment (Fully Paid): {pred_res['probability_paid']:.2%} <<<")
        print(f" >>> Probability of default (Charged Off): {pred_res['probability_default']:.2%} <<<")
    else:
        print(f" >>> Predicted Loan Amount: ${pred_res['prediction']:,.2f} <<<")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Loan Amount/Status Prediction System")
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train and save a new model')
    train_parser.add_argument('--task', type=str, choices=['regression', 'classification'], default='regression',
                              help='Task to train: regression (predict loan_amnt) or classification (predict loan_status)')
    train_parser.add_argument('--data', type=str, default='data/loan.csv', help='Path to loan.csv dataset')
    train_parser.add_argument('--model', type=str, default='model.joblib', help='Path to save the trained model')
    train_parser.add_argument('--sample-size', type=str, default='200000', 
                              help='Number of rows to sample (e.g. 200000). Set to "all" or "0" to use the entire dataset.')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict loan details for a given set of parameters')
    predict_parser.add_argument('--model', type=str, default='model.joblib', help='Path to load the trained model')
    predict_parser.add_argument('--interactive', action='store_true', help='Collect input interactively')
    predict_parser.add_argument('--input-json', type=str, help='Path to a JSON file or raw JSON string with feature inputs')
    
    # Add individual features as optional CLI flags for prediction
    for feat in NUMERIC_FEATURES + ['loan_amnt']:
        predict_parser.add_argument(f'--{feat.replace("_", "-")}', type=float, help=f'Numeric feature: {feat}')
    for feat in CATEGORICAL_FEATURES:
        predict_parser.add_argument(f'--{feat.replace("_", "-")}', type=str, help=f'Categorical feature: {feat}')
        
    args = parser.parse_args()
    
    if args.command == 'train':
        # Parse sample size
        s_size = args.sample_size.lower().strip()
        if s_size in ('all', '0', 'none'):
            sample_size = None
        else:
            try:
                sample_size = int(s_size)
            except ValueError:
                print(f"[!] Error: Invalid sample size '{args.sample_size}'. Use integer or 'all'.")
                sys.exit(1)
                
        # Train model
        model, metrics = build_loan_prediction_system(args.data, args.task, sample_size)
        
        # Save model
        print(f"[*] Saving model pipeline to {args.model}...")
        try:
            joblib.dump(model, args.model)
            print("[+] Model successfully saved!")
        except Exception as e:
            print(f"[!] Error saving model: {e}")
            sys.exit(1)
            
    elif args.command == 'predict':
        print(f"[*] Loading model pipeline from {args.model}...")
        if not os.path.exists(args.model):
            print(f"[!] Error: Model file '{args.model}' not found. Please train a model first.")
            sys.exit(1)
            
        try:
            model = load_model_safe(args.model)
            print("[+] Model loaded successfully.")
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            sys.exit(1)
            
        if args.interactive:
            interactive_prediction(model)
        elif args.input_json:
            # Parse JSON file or string
            try:
                if os.path.exists(args.input_json):
                    with open(args.input_json, 'r') as f:
                        inputs = json.load(f)
                else:
                    inputs = json.loads(args.input_json)
            except Exception as e:
                print(f"[!] Error parsing JSON input: {e}")
                sys.exit(1)
            task = getattr(model, 'task_type', 'regression')
            pred_res = predict_single(model, inputs, task=task)
            if task == 'classification':
                status_label = "Fully Paid" if pred_res['prediction'] == 1 else "Charged Off / Default"
                print(f"Predicted Loan Status: {status_label}")
                print(f"Probability of Repayment: {pred_res['probability_paid']:.2%}")
                print(f"Probability of Default: {pred_res['probability_default']:.2%}")
            else:
                print(f"Predicted Loan Amount: ${pred_res['prediction']:,.2f}")
        else:
            # Try to build inputs from individual command line flags
            inputs = {}
            missing_features = []
            
            # Task type
            task = getattr(model, 'task_type', 'regression')
            
            # Features list based on task
            num_features = NUMERIC_FEATURES + ['loan_amnt'] if task == 'classification' else NUMERIC_FEATURES
            
            # Numeric inputs
            for feat in num_features:
                cli_name = feat.replace("_", "-")
                val = getattr(args, cli_name.replace("-", "_"))
                if val is not None:
                    inputs[feat] = val
                else:
                    missing_features.append(f"--{cli_name}")
                    
            # Categorical inputs
            for feat in CATEGORICAL_FEATURES:
                cli_name = feat.replace("_", "-")
                val = getattr(args, cli_name.replace("-", "_"))
                if val is not None:
                    inputs[feat] = val
                else:
                    missing_features.append(f"--{cli_name}")
                    
            if missing_features:
                print("[!] Error: Missing required features for prediction.")
                print(f"Missing flags: {', '.join(missing_features)}")
                print("Please provide all features or run prediction with --interactive flag.")
                sys.exit(1)
                
            pred_res = predict_single(model, inputs, task=task)
            if task == 'classification':
                status_label = "Fully Paid" if pred_res['prediction'] == 1 else "Charged Off / Default"
                print(f"Predicted Loan Status: {status_label}")
                print(f"Probability of Repayment: {pred_res['probability_paid']:.2%}")
                print(f"Probability of Default: {pred_res['probability_default']:.2%}")
            else:
                print(f"Predicted Loan Amount: ${pred_res['prediction']:,.2f}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()