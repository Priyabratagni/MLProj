from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib, numpy as np
import os
import json
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('--test-dir', type=str, required=True, help='Directory containing test data')
parser.add_argument('--models-dir', type=str, required=True, help='Directory containing models')
parser.add_argument('--output', type=str, required=True, help='Output directory for results')
args = parser.parse_args()

# Use the parsed arguments
test_dir = args.test_dir
models_dir = args.models_dir
output_dir = args.output

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

if __name__=='__main__':
    X_test = np.load(os.path.join(test_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'), allow_pickle=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name in ['random_forest','logistic_regression']:
        model = joblib.load(os.path.join(models_dir, f'{name}.pkl'))
        results[name] = evaluate(model, X_test, y_test)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
