"""
ENSEMBLE MODELS: RandomForest + Self-Organizing Map (SOM)
Complementary approaches to deep learning for exoplanet detection

RandomForest: Feature-based classification
SOM: Unsupervised clustering and anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class ExoplanetFeatureExtractor:
    """
    Extract engineered features from light curves for traditional ML
    Features capture transit characteristics without deep learning
    """
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, light_curve):
        """
        Extract comprehensive features from a single light curve
        
        Returns: Feature vector (numpy array)
        """
        features = {}
        lc = light_curve.flatten() if len(light_curve.shape) > 1 else light_curve
        
        # ========== STATISTICAL FEATURES ==========
        features['mean'] = np.mean(lc)
        features['std'] = np.std(lc)
        features['median'] = np.median(lc)
        features['mad'] = np.median(np.abs(lc - np.median(lc)))  # Median Absolute Deviation
        features['min'] = np.min(lc)
        features['max'] = np.max(lc)
        features['range'] = features['max'] - features['min']
        features['skewness'] = stats.skew(lc)
        features['kurtosis'] = stats.kurtosis(lc)
        
        # Percentiles
        features['q25'] = np.percentile(lc, 25)
        features['q75'] = np.percentile(lc, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # ========== TRANSIT-SPECIFIC FEATURES ==========
        # Count deep dips (potential transits)
        threshold = np.median(lc) - 2 * features['mad']
        features['n_deep_dips'] = np.sum(lc < threshold)
        
        # Find peaks (inverted for dips)
        inverted_lc = -lc
        peaks, properties = find_peaks(inverted_lc, prominence=0.5, distance=10)
        features['n_peaks'] = len(peaks)
        
        if len(peaks) > 0:
            features['peak_prominence_mean'] = np.mean(properties['prominences'])
            features['peak_prominence_max'] = np.max(properties['prominences'])
        else:
            features['peak_prominence_mean'] = 0
            features['peak_prominence_max'] = 0
        
        # Transit depth estimate
        features['transit_depth'] = features['median'] - features['min']
        
        # ========== TEMPORAL FEATURES ==========
        # First differences (rate of change)
        diff = np.diff(lc)
        features['diff_mean'] = np.mean(np.abs(diff))
        features['diff_std'] = np.std(diff)
        features['diff_max'] = np.max(np.abs(diff))
        
        # Second differences (acceleration)
        diff2 = np.diff(diff)
        features['diff2_mean'] = np.mean(np.abs(diff2))
        features['diff2_std'] = np.std(diff2)
        
        # ========== FREQUENCY DOMAIN FEATURES ==========
        # FFT for periodicity detection
        fft = np.fft.fft(lc)
        fft_power = np.abs(fft[:len(fft)//2])
        features['fft_max_power'] = np.max(fft_power)
        features['fft_mean_power'] = np.mean(fft_power)
        features['fft_std_power'] = np.std(fft_power)
        
        # Dominant frequency
        if len(fft_power) > 0:
            features['dominant_freq_idx'] = np.argmax(fft_power)
        else:
            features['dominant_freq_idx'] = 0
        
        # ========== SHAPE FEATURES ==========
        # Flatness (how flat is the baseline)
        sorted_lc = np.sort(lc)
        top_80_percent = sorted_lc[int(0.2*len(sorted_lc)):]
        features['flatness'] = np.std(top_80_percent)
        
        # Asymmetry
        left_half = lc[:len(lc)//2]
        right_half = lc[len(lc)//2:]
        features['asymmetry'] = np.abs(np.mean(left_half) - np.mean(right_half))
        
        self.feature_names = list(features.keys())
        return np.array(list(features.values()))
    
    def extract_batch(self, light_curves, verbose=True):
        """Extract features from multiple light curves"""
        features_list = []
        
        for i, lc in enumerate(light_curves):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Extracted features from {i+1}/{len(light_curves)} light curves")
            
            features = self.extract_features(lc)
            features_list.append(features)
        
        X_features = np.array(features_list)
        
        # Handle NaN/inf
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return X_features


class RandomForestExoplanetDetector:
    """
    RandomForest classifier for exoplanet detection
    Uses engineered features instead of raw light curves
    """
    
    def __init__(self, 
                 n_estimators=200, 
                 max_depth=30,
                 model_path='models/random_forest.pkl'):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
            verbose=1,
            class_weight='balanced'
        )
        self.feature_extractor = ExoplanetFeatureExtractor()
        self.model_path = model_path
        self.feature_names = None
    
    def train(self, X_train, y_train):
        """
        Train RandomForest on light curves
        
        Args:
            X_train: Training light curves
            y_train: Training labels
        """
        print("\n" + "=" * 70)
        print("TRAINING RANDOMFOREST CLASSIFIER")
        print("=" * 70)
        
        # Extract features
        print("Extracting features from training data...")
        X_features = self.feature_extractor.extract_batch(X_train)
        self.feature_names = self.feature_extractor.feature_names
        
        print(f"Feature shape: {X_features.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        # Train
        print("\nTraining RandomForest...")
        self.model.fit(X_features, y_train)
        
        print("=" * 70)
        print("✓ RandomForest training completed!")
        print("=" * 70 + "\n")
    
    def predict(self, X_test):
        """Make predictions"""
        X_features = self.feature_extractor.extract_batch(X_test, verbose=False)
        predictions = self.model.predict(X_features)
        probabilities = self.model.predict_proba(X_features)[:, 1]
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        predictions, probabilities = self.predict(X_test)
        
        print("\n" + "=" * 70)
        print("RANDOMFOREST TEST RESULTS")
        print("=" * 70)
        
        print(classification_report(y_test, predictions, 
                                   target_names=['No Planet', 'Exoplanet']))
        
        auc = roc_auc_score(y_test, probabilities)
        print(f"AUC-ROC: {auc:.4f}")
        print("=" * 70 + "\n")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'auc': auc
        }
    
    def get_feature_importance(self, top_n=15):
        """Get most important features"""
        if self.feature_names is None:
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print("\n" + "=" * 70)
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print("=" * 70)
        
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {self.feature_names[idx]:25s} : {importances[idx]:.4f}")
        
        print("=" * 70 + "\n")
        
        return list(zip([self.feature_names[i] for i in indices], 
                       importances[indices]))
    
    def save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_extractor': self.feature_extractor,
                'feature_names': self.feature_names
            }, f)
        
        print(f"✓ RandomForest saved to {self.model_path}")
    
    def load_model(self):
        """Load saved model"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_extractor = data['feature_extractor']
        self.feature_names = data['feature_names']
        
        print(f"✓ RandomForest loaded from {self.model_path}")


class SelfOrganizingMap:
    """
    Self-Organizing Map (SOM) for unsupervised exoplanet detection
    Useful for anomaly detection and clustering
    """
    
    def __init__(self, map_size=(10, 10), input_dim=30, learning_rate=0.5):
        """
        Args:
            map_size: Size of SOM grid (height, width)
            input_dim: Dimension of input vectors
            learning_rate: Initial learning rate
        """
        self.map_size = map_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        self.weights = np.random.randn(map_size[0], map_size[1], input_dim)
        self.trained = False
    
    def _euclidean_distance(self, x, w):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x - w) ** 2, axis=-1))
    
    def _find_bmu(self, x):
        """Find Best Matching Unit (BMU) for input x"""
        distances = self._euclidean_distance(x, self.weights)
        bmu_idx = np.unravel_index(np.argmin(distances), self.map_size)
        return bmu_idx
    
    def _neighborhood_function(self, bmu_idx, sigma):
        """Calculate neighborhood influence"""
        y, x = np.meshgrid(range(self.map_size[1]), range(self.map_size[0]))
        d = np.sqrt((x - bmu_idx[0])**2 + (y - bmu_idx[1])**2)
        return np.exp(-d**2 / (2 * sigma**2))
    
    def train(self, X, epochs=100):
        """
        Train SOM on data
        
        Args:
            X: Training data (n_samples, features)
            epochs: Number of training epochs
        """
        print("\n" + "=" * 70)
        print("TRAINING SELF-ORGANIZING MAP (SOM)")
        print("=" * 70)
        print(f"Map size: {self.map_size}")
        print(f"Input dimension: {self.input_dim}")
        print(f"Epochs: {epochs}")
        
        n_samples = len(X)
        initial_radius = max(self.map_size) / 2
        
        for epoch in range(epochs):
            # Decay parameters
            sigma = initial_radius * np.exp(-epoch / epochs)
            lr = self.learning_rate * np.exp(-epoch / epochs)
            
            # Random training order
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x = X[idx]
                
                # Find BMU
                bmu_idx = self._find_bmu(x)
                
                # Update weights
                neighborhood = self._neighborhood_function(bmu_idx, sigma)
                
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        self.weights[i, j] += (lr * neighborhood[i, j] * 
                                              (x - self.weights[i, j]))
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} - σ={sigma:.3f}, lr={lr:.4f}")
        
        self.trained = True
        print("=" * 70)
        print("✓ SOM training completed!")
        print("=" * 70 + "\n")
    
    def predict_anomaly(self, X, threshold_percentile=95):
        """
        Detect anomalies based on distance to BMU
        
        Args:
            X: Data to analyze
            threshold_percentile: Percentile for anomaly threshold
        
        Returns:
            anomaly_scores: Distance to BMU (higher = more anomalous)
            is_anomaly: Boolean array indicating anomalies
        """
        if not self.trained:
            raise ValueError("SOM must be trained first!")
        
        anomaly_scores = []
        
        for x in X:
            bmu_idx = self._find_bmu(x)
            bmu_weights = self.weights[bmu_idx]
            distance = np.linalg.norm(x - bmu_weights)
            anomaly_scores.append(distance)
        
        anomaly_scores = np.array(anomaly_scores)
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        is_anomaly = anomaly_scores > threshold
        
        return anomaly_scores, is_anomaly
    
    def get_cluster_assignments(self, X):
        """Assign data points to SOM clusters"""
        cluster_assignments = []
        
        for x in X:
            bmu_idx = self._find_bmu(x)
            # Convert 2D index to 1D cluster ID
            cluster_id = bmu_idx[0] * self.map_size[1] + bmu_idx[1]
            cluster_assignments.append(cluster_id)
        
        return np.array(cluster_assignments)


if __name__ == "__main__":
    print("Ensemble Models for Exoplanet Detection")
    print("=" * 70)
    print("\n1. RandomForest: Feature-based classification")
    print("   - Extracts 30+ engineered features")
    print("   - Robust to overfitting")
    print("   - Fast inference")
    print("   - Interpretable (feature importance)")
    
    print("\n2. Self-Organizing Map (SOM): Unsupervised detection")
    print("   - Clusters similar light curves")
    print("   - Anomaly detection")
    print("   - No labels required")
    print("   - Discovers hidden patterns")
    
    print("\n=" * 70)
    print("Use with: python3 ensemble_training.py")
