import argparse
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def main(train_dir, models_dir):
    # Load train data
    X_train = np.load(os.path.join(train_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'), allow_pickle=True)

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Save Random Forest
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))

    # Train Logistic Regression
    lr_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=2000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    # Save Logistic Regression
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))

    print(f"Models trained and saved inside '{models_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--models-dir', type=str, required=True)
    args = parser.parse_args()
    main(args.train_dir, args.models_dir)
