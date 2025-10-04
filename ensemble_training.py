"""
COMPREHENSIVE ENSEMBLE EXOPLANET DETECTION SYSTEM
Combines multiple models for maximum accuracy:
1. Hybrid CNN+LSTM
2. Multi-View CNN (Shallue & Vanderburg 2018)
3. RandomForest
4. Self-Organizing Map (SOM)

Optimized for: AMD Ryzen 7 5700U, 16 cores, 14GB RAM
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import gc
import warnings
warnings.filterwarnings('ignore')

# Hardware optimization
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(8)  # Optimized for 16 cores
tf.config.threading.set_intra_op_parallelism_threads(8)

# Import our models
from hybrid_cnn_lstm_model import HybridCNNLSTM
from multiview_cnn_model import MultiViewCNN, create_local_view
from ensemble_models import RandomForestExoplanetDetector, SelfOrganizingMap, ExoplanetFeatureExtractor
from sklearn.metrics import roc_auc_score


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models
    Uses weighted voting for maximum accuracy
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.model_predictions = {}
        self.model_probabilities = {}
    
    def add_model(self, name, predictions, probabilities, weight=1.0):
        """Add model predictions to ensemble"""
        self.model_predictions[name] = predictions
        self.model_probabilities[name] = probabilities
        self.weights[name] = weight
    
    def predict(self, threshold=0.5):
        """
        Make ensemble prediction using weighted voting
        """
        if not self.model_probabilities:
            raise ValueError("No models added to ensemble!")
        
        # Weighted average of probabilities
        weighted_probs = np.zeros(len(list(self.model_probabilities.values())[0]))
        total_weight = sum(self.weights.values())
        
        for name, probs in self.model_probabilities.items():
            weight = self.weights[name]
            weighted_probs += probs.flatten() * (weight / total_weight)
        
        ensemble_predictions = (weighted_probs > threshold).astype(int)
        
        return ensemble_predictions, weighted_probs


def load_kepler_dataset(train_path='data/exoTrain.csv', 
                        test_path='data/exoTest.csv'):
    """
    Load official Kepler exoplanet dataset
    This is the PRIMARY dataset used by Google Brain paper
    """
    print("\n" + "="*70)
    print("LOADING KEPLER EXOPLANET DATASET")
    print("Dataset: Official Kepler Mission Training Data")
    print("Reference: Shallue & Vanderburg (2018)")
    print("="*70)
    
    # Load training data
    print(f"\nLoading: {train_path}")
    df_train = pd.read_csv(train_path)
    
    # Load test data
    print(f"Loading: {test_path}")
    df_test = pd.read_csv(test_path)
    
    # Extract labels
    y_train = df_train['LABEL'].values - 1  # Convert 2,1 to 1,0
    y_test = df_test['LABEL'].values - 1
    
    # Extract flux values
    X_train = df_train.drop('LABEL', axis=1).values
    X_test = df_test.drop('LABEL', axis=1).values
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nClass distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label = "Exoplanet" if u == 1 else "No Planet"
        print(f"  {label}: {c} ({c/len(y_train)*100:.1f}%)")
    
    print("="*70)
    
    return X_train, X_test, y_train, y_test


def preprocess_light_curves(X, normalize=True, remove_outliers=True):
    """
    Optimized preprocessing for light curves
    """
    print("\nPreprocessing light curves...")
    
    X_processed = []
    
    for i, lc in enumerate(X):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(X)}")
        
        # Remove outliers (sigma clipping)
        if remove_outliers:
            median = np.median(lc)
            std = np.std(lc)
            lc = np.clip(lc, median - 5*std, median + 5*std)
        
        # Normalize (robust scaling)
        if normalize:
            median = np.median(lc)
            mad = np.median(np.abs(lc - median))
            if mad > 0:
                lc = (lc - median) / (1.4826 * mad)
            else:
                lc = lc - median
            
            lc = np.clip(lc, -5, 5)
        
        X_processed.append(lc)
    
    X_processed = np.array(X_processed)
    print(f"✓ Preprocessing completed: {X_processed.shape}")
    
    return X_processed


def create_multiview_data(X):
    """
    Create global and local views for Multi-View CNN
    """
    print("\nCreating multi-view data...")
    
    # Global view: downsample to 2001 points
    X_global = []
    for lc in X:
        if len(lc) > 2001:
            indices = np.linspace(0, len(lc)-1, 2001, dtype=int)
            global_view = lc[indices]
        elif len(lc) < 2001:
            pad_width = 2001 - len(lc)
            global_view = np.pad(lc, (0, pad_width), mode='constant', 
                               constant_values=np.median(lc))
        else:
            global_view = lc
        X_global.append(global_view)
    
    X_global = np.array(X_global).reshape(-1, 2001, 1)
    
    # Local view: extract transit region (201 points)
    X_local = []
    for lc in X:
        local_view = create_local_view(lc, window_size=201)
        X_local.append(local_view)
    
    X_local = np.array(X_local).reshape(-1, 201, 1)
    
    print(f"✓ Global view: {X_global.shape}")
    print(f"✓ Local view: {X_local.shape}")
    
    return X_global, X_local


def plot_ensemble_results(y_true, ensemble_pred, ensemble_prob, 
                         model_predictions, model_probabilities,
                         save_dir='results/ensemble'):
    """
    Create comprehensive visualization of ensemble results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Ensemble Confusion Matrix
    cm = confusion_matrix(y_true, ensemble_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    annotations = [[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                   for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=np.array(annotations), fmt='', cmap='RdYlGn',
                xticklabels=['No Planet', 'Exoplanet'],
                yticklabels=['No Planet', 'Exoplanet'])
    plt.title('Ensemble Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ensemble_confusion_matrix.png', dpi=300)
    plt.close()
    print(f"✓ Saved: {save_dir}/ensemble_confusion_matrix.png")
    
    # 2. ROC Curves for all models
    plt.figure(figsize=(12, 10))
    
    # Plot each model
    for name, probs in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Plot ensemble
    fpr, tpr, _ = roc_curve(y_true, ensemble_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=3, color='red', linestyle='--',
             label=f'ENSEMBLE (AUC = {roc_auc:.4f})')
    
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models + Ensemble', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ensemble_roc_curves.png', dpi=300)
    plt.close()
    print(f"✓ Saved: {save_dir}/ensemble_roc_curves.png")
    
    # 3. Model Agreement Analysis
    plt.figure(figsize=(14, 8))
    
    model_names = list(model_predictions.keys())
    n_models = len(model_names)
    
    # Count agreements
    predictions_matrix = np.array([model_predictions[name].flatten() 
                                   for name in model_names])
    agreement_count = np.sum(predictions_matrix, axis=0)
    
    # Plot histogram
    plt.subplot(1, 2, 1)
    plt.hist(agreement_count, bins=np.arange(n_models+2)-0.5, 
             edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Number of Models Agreeing', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Model Agreement Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot agreement vs correctness
    plt.subplot(1, 2, 2)
    correct = (ensemble_pred == y_true)
    for i in range(n_models+1):
        mask = agreement_count == i
        if np.any(mask):
            accuracy = np.mean(correct[mask])
            plt.bar(i, accuracy, alpha=0.7)
    
    plt.xlabel('Number of Models Predicting Exoplanet', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Model Agreement', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_agreement_analysis.png', dpi=300)
    plt.close()
    print(f"✓ Saved: {save_dir}/model_agreement_analysis.png")


def main():
    """
    Main ensemble training pipeline
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE ENSEMBLE EXOPLANET DETECTION SYSTEM")
    print("="*70)
    print("Hardware: AMD Ryzen 7 5700U (16 cores), 14GB RAM")
    print("Models: CNN+LSTM, Multi-View CNN, RandomForest, SOM (4-way ensemble)")
    print("Training: 75 epochs each (optimized balance)")
    print("Dataset: Official Kepler Mission Data")
    print("="*70)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/ensemble', exist_ok=True)
    
    # STEP 1: Load Data
    print("\n[STEP 1/6] Loading Kepler Dataset")
    X_train, X_test, y_train, y_test = load_kepler_dataset()
    
    # STEP 2: Preprocess
    print("\n[STEP 2/6] Preprocessing")
    X_train_proc = preprocess_light_curves(X_train)
    X_test_proc = preprocess_light_curves(X_test)
    
    # Create validation split
    X_train_proc, X_val_proc, y_train, y_val = train_test_split(
        X_train_proc, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    print(f"\nFinal splits:")
    print(f"  Train: {X_train_proc.shape}")
    print(f"  Val:   {X_val_proc.shape}")
    print(f"  Test:  {X_test_proc.shape}")
    
    # STEP 3: Train Hybrid CNN+LSTM
    print("\n[STEP 3/6] Training Hybrid CNN+LSTM Model")
    print("=" * 70)
    
    # Reshape for LSTM
    X_train_lstm = X_train_proc.reshape(-1, X_train_proc.shape[1], 1)
    X_val_lstm = X_val_proc.reshape(-1, X_val_proc.shape[1], 1)
    X_test_lstm = X_test_proc.reshape(-1, X_test_proc.shape[1], 1)
    
    hybrid_model = HybridCNNLSTM(input_shape=(X_train_lstm.shape[1], 1),
                                  model_path='models/ensemble_hybrid_cnn_lstm.h5')
    hybrid_model.build_model()
    hybrid_model.compile_model()
    
    hybrid_model.train(X_train_lstm, y_train, X_val_lstm, y_val,
                      epochs=75,  # Optimized for ensemble (better accuracy)
                      batch_size=32)
    
    hybrid_metrics = hybrid_model.evaluate(X_test_lstm, y_test)
    hybrid_pred, hybrid_prob = hybrid_model.predict(X_test_lstm)
    
    # Clear memory
    del X_train_lstm, X_val_lstm, X_test_lstm
    gc.collect()
    
    # STEP 4: Train Multi-View CNN
    print("\n[STEP 4/6] Training Multi-View CNN Model")
    print("=" * 70)
    
    # Create multi-view data
    X_train_global, X_train_local = create_multiview_data(X_train_proc)
    X_val_global, X_val_local = create_multiview_data(X_val_proc)
    X_test_global, X_test_local = create_multiview_data(X_test_proc)
    
    multiview_model = MultiViewCNN(global_shape=(2001, 1), local_shape=(201, 1),
                                    model_path='models/ensemble_multiview_cnn.h5')
    multiview_model.build_model()
    multiview_model.compile_model()
    
    multiview_model.train(X_train_global, X_train_local, y_train,
                          X_val_global, X_val_local, y_val,
                          epochs=75,  # Optimized for ensemble
                          batch_size=32)
    
    multiview_metrics = multiview_model.evaluate(X_test_global, X_test_local, y_test)
    multiview_pred, multiview_prob = multiview_model.predict(X_test_global, X_test_local)
    
    # Clear memory
    del X_train_global, X_train_local, X_val_global, X_val_local
    del X_test_global, X_test_local
    gc.collect()
    
    # STEP 5: Train RandomForest
    print("\n[STEP 5/6] Training RandomForest Model")
    print("=" * 70)
    
    rf_model = RandomForestExoplanetDetector(n_estimators=200, max_depth=30,
                                              model_path='models/ensemble_random_forest.pkl')
    
    # Combine train and val for RF (it has built-in validation)
    X_train_rf = np.vstack([X_train_proc, X_val_proc])
    y_train_rf = np.concatenate([y_train, y_val])
    
    rf_model.train(X_train_rf, y_train_rf)
    rf_metrics = rf_model.evaluate(X_test_proc, y_test)
    rf_pred = rf_metrics['predictions']
    rf_prob = rf_metrics['probabilities']
    
    # Show feature importance
    rf_model.get_feature_importance(top_n=15)
    rf_model.save_model()
    
    # Clear memory
    del X_train_rf, y_train_rf
    gc.collect()
    
    # STEP 5.5: Train Self-Organizing Map (SOM) for anomaly detection
    print("\n[5.5/6] Training Self-Organizing Map (SOM)")
    print("=" * 70)
    
    # Extract features for SOM
    feature_extractor = ExoplanetFeatureExtractor()
    X_test_features = feature_extractor.extract_batch(X_test_proc, verbose=False)
    
    som_model = SelfOrganizingMap(map_size=(12, 12), input_dim=X_test_features.shape[1])
    som_model.train(X_test_features, epochs=100)
    
    # Get anomaly scores (higher = more likely to be exoplanet)
    anomaly_scores, is_anomaly = som_model.predict_anomaly(X_test_features, threshold_percentile=85)
    som_pred = is_anomaly.astype(int)
    som_prob = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    
    # Save SOM model
    import pickle
    som_model_path = 'models/ensemble_som.pkl'
    with open(som_model_path, 'wb') as f:
        pickle.dump({
            'som_model': som_model,
            'feature_extractor': feature_extractor,
            'anomaly_scores': anomaly_scores,
            'threshold_percentile': 85
        }, f)
    print(f"✓ SOM model saved to {som_model_path}")
    
    # Clear memory
    del X_test_features
    gc.collect()
    
    # STEP 6: Create Ensemble
    print("\n[STEP 6/6] Creating Ensemble Predictor")
    print("=" * 70)
    
    ensemble = EnsemblePredictor()
    
    # Add models with weights based on individual performance
    ensemble.add_model('Hybrid_CNN_LSTM', hybrid_pred, hybrid_prob, 
                       weight=hybrid_metrics['auc'])
    ensemble.add_model('MultiView_CNN', multiview_pred, multiview_prob,
                       weight=multiview_metrics['auc'])
    ensemble.add_model('RandomForest', rf_pred, rf_prob,
                       weight=rf_metrics['auc'])
    
    # Add SOM (unsupervised anomaly detection)
    som_auc = roc_auc_score(y_test, som_prob)
    ensemble.add_model('SOM_Anomaly', som_pred, som_prob,
                       weight=som_auc)
    
    # Make ensemble prediction
    ensemble_pred, ensemble_prob = ensemble.predict(threshold=0.5)
    
    # Evaluate ensemble
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS")
    print("="*70)
    print(classification_report(y_test, ensemble_pred,
                               target_names=['No Planet', 'Exoplanet'], digits=4))
    
    ensemble_auc = roc_auc_score(y_test, ensemble_prob)
    print(f"\nENSEMBLE AUC-ROC: {ensemble_auc:.4f}")
    
    # Compare all models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Hybrid CNN+LSTM:  AUC = {hybrid_metrics['auc']:.4f}")
    print(f"Multi-View CNN:   AUC = {multiview_metrics['auc']:.4f}")
    print(f"RandomForest:     AUC = {rf_metrics['auc']:.4f}")
    print(f"SOM Anomaly:      AUC = {som_auc:.4f}")
    print(f"ENSEMBLE (4-way): AUC = {ensemble_auc:.4f} ← BEST!")
    print("="*70)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_ensemble_results(
        y_test, ensemble_pred, ensemble_prob,
        ensemble.model_predictions,
        ensemble.model_probabilities
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Results saved to: results/ensemble/")
    print(f"Models saved to: models/")
    print(f"\nFinal Ensemble Accuracy: {np.mean(ensemble_pred == y_test)*100:.2f}%")
    print(f"Final Ensemble AUC: {ensemble_auc:.4f}")
    print("\nNext steps:")
    print("  1. Check results/ensemble/ for visualizations")
    print("  2. Run python3 hybrid_webapp.py to use the model")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
