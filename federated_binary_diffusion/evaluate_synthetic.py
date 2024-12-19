import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_synthetic_data(original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """Evaluate the quality of synthetic data"""
    
    # 1. Statistical Similarity
    def compare_distributions(col):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(original_data[col], label='Original')
        plt.title(f'Original {col} Distribution')
        
        plt.subplot(1, 2, 2)
        sns.histplot(synthetic_data[col], label='Synthetic')
        plt.title(f'Synthetic {col} Distribution')
        
        plt.tight_layout()
        plt.savefig(f'./results/distribution_{col}.png')
        plt.close()
    
    # Compare distributions for important columns
    for col in ['Age', 'Number of sexual partners', 'First sexual intercourse']:
        compare_distributions(col)
    
    # 2. Privacy Assessment
    def check_privacy():
        """Check if synthetic data contains exact copies from original data"""
        duplicates = 0
        for _, synthetic_row in synthetic_data.iterrows():
            if (original_data == synthetic_row).all(1).any():
                duplicates += 1
        return duplicates
    
    privacy_score = check_privacy()
    print(f"Number of exact duplicates: {privacy_score}")
    
    # 3. Utility Assessment
    # Train model on synthetic, test on real
    X_real = original_data.drop('Biopsy', axis=1)
    y_real = original_data['Biopsy']
    X_synthetic = synthetic_data.drop('Biopsy', axis=1)
    y_synthetic = synthetic_data['Biopsy']
    
    # Split real data
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    
    # Train on synthetic, test on real
    rf_synthetic = RandomForestClassifier(random_state=42)
    rf_synthetic.fit(X_synthetic, y_synthetic)
    synthetic_score = rf_synthetic.score(X_real_test, y_real_test)
    
    # Train on real, test on real (baseline)
    rf_real = RandomForestClassifier(random_state=42)
    rf_real.fit(X_real_train, y_real_train)
    real_score = rf_real.score(X_real_test, y_real_test)
    
    print("\nUtility Scores:")
    print(f"Model trained on real data: {real_score:.4f}")
    print(f"Model trained on synthetic data: {synthetic_score:.4f}")
    
    return {
        'privacy_score': privacy_score,
        'real_score': real_score,
        'synthetic_score': synthetic_score
    }

if __name__ == "__main__":
    original_data = pd.read_csv("../data/cervical_train.csv")
    synthetic_data = pd.read_csv("./results/synthetic_data.csv")
    
    results = evaluate_synthetic_data(original_data, synthetic_data)
    
    # Save results
    with open('./results/evaluation_report.txt', 'w') as f:
        f.write("Synthetic Data Evaluation Report\n")
        f.write("==============================\n\n")
        f.write(f"Privacy Score (lower is better): {results['privacy_score']}\n")
        f.write(f"Real Data Model Score: {results['real_score']:.4f}\n")
        f.write(f"Synthetic Data Model Score: {results['synthetic_score']:.4f}\n")
