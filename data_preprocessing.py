import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json

def main(input_file, output_dir):
    # Load dataset
    df = pd.read_csv(input_file)
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Simple encoding if needed (optional, based on columns)
    X = pd.get_dummies(X, drop_first=True)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    print(f"Data splits saved under '{output_dir}/' successfully!")

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Dropout', 'Enrolled', 'Graduate'], yticklabels=['Dropout', 'Enrolled', 'Graduate'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.show()

# Function to plot classification report metrics
def plot_classification_report(report, model_name, output_dir):
    categories = ['Dropout', 'Enrolled', 'Graduate']
    precision = [report[cat]['precision'] for cat in categories]
    recall = [report[cat]['recall'] for cat in categories]
    f1_score = [report[cat]['f1-score'] for cat in categories]

    x = np.arange(len(categories))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1-Score')

    plt.xlabel('Categories')
    plt.ylabel('Scores')
    plt.title(f'Classification Report for {model_name}')
    plt.xticks(x, categories)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_classification_report.png'))
    plt.show()

# Load results from JSON
with open('results/metrics.json', 'r') as f:
    results = json.load(f)

# Plot results for each model
for model_name, metrics in results.items():
    plot_confusion_matrix(metrics['confusion_matrix'], model_name, 'results')
    plot_classification_report(metrics['report'], model_name, 'results')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save data splits.")
    args = parser.parse_args()

    main(args.input, args.output)
