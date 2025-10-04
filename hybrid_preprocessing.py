"""
ENHANCED EXOPLANET DATA PREPROCESSOR
Optimized for both CNN and Hybrid CNN+LSTM models
Includes advanced normalization and data handling
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import pickle
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d


class ExoplanetDataPreprocessor:
    """
    Advanced preprocessor for exoplanet light curve data
    Handles multiple dataset formats and normalization strategies
    """
    
    def __init__(self, data_dir='data', normalization_method='robust'):
        """
        Args:
            data_dir: Directory containing data files
            normalization_method: 'robust', 'standard', or 'minmax'
        """
        self.data_dir = data_dir
        self.normalization_method = normalization_method
        self.scaler = RobustScaler() if normalization_method == 'robust' else StandardScaler()
        self.input_length = None
        
    def load_csv_dataset(self, filepath):
        """Load CSV format dataset"""
        print(f"Loading {filepath}...")
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_light_curves(self, df, label_column=None):
        """
        Extract light curves and labels from DataFrame
        Auto-detects label column if not specified
        """
        # Auto-detect label column
        if label_column is None:
            label_cols = [col for col in df.columns if any(
                keyword in col.lower() for keyword in 
                ['label', 'disposition', 'class', 'target', 'koi']
            )]
            label_column = label_cols[0] if label_cols else None
        
        if label_column and label_column in df.columns:
            print(f"Using label column: {label_column}")
            y = df[label_column].values
            X = df.drop(columns=[label_column]).values
            
            # Convert string labels to binary
            if y.dtype == 'object':
                label_map = {
                    'CONFIRMED': 1, 'CANDIDATE': 1, 'PC': 1, '2': 1,
                    'FALSE POSITIVE': 0, 'FP': 0, 'NOT PLANETARY': 0, '1': 0
                }
                y = np.array([label_map.get(str(v).upper(), int(v) if str(v).isdigit() else 0) 
                             for v in y])
            else:
                y = (y > 1).astype(int) if np.max(y) > 1 else y.astype(int)
            
            return X, y
        else:
            print("No label column found - returning features only")
            return df.values, None
    
    def remove_outliers(self, light_curve, sigma=5):
        """Remove outliers using sigma clipping"""
        median = np.median(light_curve)
        std = np.std(light_curve)
        
        # Clip outliers
        clipped = np.clip(light_curve, median - sigma*std, median + sigma*std)
        return clipped
    
    def apply_median_filter(self, light_curve, kernel_size=5):
        """Apply median filter to reduce noise"""
        if len(light_curve) < kernel_size:
            return light_curve
        return medfilt(light_curve, kernel_size=kernel_size)
    
    def apply_gaussian_smoothing(self, light_curve, sigma=2):
        """Apply Gaussian smoothing"""
        return gaussian_filter1d(light_curve, sigma=sigma)
    
    def normalize_light_curve(self, light_curve, method='robust'):
        """
        Normalize light curve using various methods
        
        Methods:
        - 'robust': Median and MAD (best for astronomical data)
        - 'standard': Mean and std (z-score)
        - 'minmax': Scale to [0, 1]
        """
        # Handle NaN/inf values
        light_curve = np.nan_to_num(light_curve, nan=np.nanmedian(light_curve))
        
        if method == 'robust':
            # Robust normalization (median and MAD)
            median = np.median(light_curve)
            mad = np.median(np.abs(light_curve - median))
            
            if mad > 0:
                light_curve = (light_curve - median) / (1.4826 * mad)
            else:
                light_curve = light_curve - median
                
        elif method == 'standard':
            # Standard normalization (z-score)
            mean = np.mean(light_curve)
            std = np.std(light_curve)
            
            if std > 0:
                light_curve = (light_curve - mean) / std
            else:
                light_curve = light_curve - mean
                
        elif method == 'minmax':
            # Min-Max scaling
            min_val = np.min(light_curve)
            max_val = np.max(light_curve)
            
            if max_val > min_val:
                light_curve = (light_curve - min_val) / (max_val - min_val)
        
        # Final outlier clipping
        light_curve = np.clip(light_curve, -5, 5)
        
        return light_curve
    
    def detrend_light_curve(self, light_curve, window=101):
        """
        Remove long-term trends from light curve
        Preserves short-term variations (transits)
        """
        if len(light_curve) < window:
            window = len(light_curve) // 2
            if window % 2 == 0:
                window += 1
        
        # Apply median filter to get trend
        trend = medfilt(light_curve, kernel_size=window)
        
        # Subtract trend
        detrended = light_curve - trend
        
        return detrended
    
    def pad_or_truncate(self, light_curve, target_length):
        """Ensure light curve has correct length"""
        current_length = len(light_curve)
        
        if current_length < target_length:
            # Pad with median
            pad_value = np.median(light_curve)
            padded = np.pad(light_curve, (0, target_length - current_length),
                          mode='constant', constant_values=pad_value)
            return padded
        elif current_length > target_length:
            # Truncate
            return light_curve[:target_length]
        else:
            return light_curve
    
    def preprocess_single_curve(self, light_curve, apply_detrend=True, 
                                apply_smoothing=False):
        """
        Complete preprocessing pipeline for single light curve
        
        Args:
            light_curve: Raw flux values
            apply_detrend: Remove long-term trends
            apply_smoothing: Apply Gaussian smoothing
        """
        # Remove outliers
        light_curve = self.remove_outliers(light_curve, sigma=5)
        
        # Detrend (optional, good for LSTM)
        if apply_detrend:
            light_curve = self.detrend_light_curve(light_curve)
        
        # Smooth (optional)
        if apply_smoothing:
            light_curve = self.apply_gaussian_smoothing(light_curve, sigma=2)
        
        # Normalize
        light_curve = self.normalize_light_curve(light_curve, method=self.normalization_method)
        
        return light_curve
    
    def preprocess_dataset(self, X, y=None, test_size=0.2, val_size=0.1,
                          apply_detrend=True, apply_smoothing=False):
        """
        Complete preprocessing pipeline for entire dataset
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print(f"Original data shape: {X.shape}")
        
        # Store input length
        self.input_length = X.shape[1]
        
        # Preprocess each light curve
        print("Preprocessing light curves...")
        X_processed = []
        for i, lc in enumerate(X):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(X)} light curves")
            
            processed_lc = self.preprocess_single_curve(
                lc, 
                apply_detrend=apply_detrend,
                apply_smoothing=apply_smoothing
            )
            X_processed.append(processed_lc)
        
        X_processed = np.array(X_processed)
        
        # Reshape for CNN/LSTM (samples, timesteps, features)
        X_processed = X_processed.reshape(X_processed.shape[0], X_processed.shape[1], 1)
        
        print(f"Processed data shape: {X_processed.shape}")
        
        if y is not None:
            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            print(f"\nClass distribution: {dict(zip(unique, counts))}")
            
            # Check if stratified split is possible
            min_class_count = min(counts)
            use_stratify = min_class_count >= 2
            
            if not use_stratify:
                print(f"Warning: Minimum class count is {min_class_count}")
                print("Using non-stratified split")
            
            # Split data
            try:
                if use_stratify:
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_processed, y, 
                        test_size=(test_size + val_size),
                        stratify=y, 
                        random_state=42
                    )
                    
                    # Second split
                    val_ratio = val_size / (test_size + val_size)
                    unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
                    
                    if min(counts_temp) >= 2:
                        X_val, X_test, y_val, y_test = train_test_split(
                            X_temp, y_temp,
                            test_size=(1 - val_ratio),
                            stratify=y_temp,
                            random_state=42
                        )
                    else:
                        X_val, X_test, y_val, y_test = train_test_split(
                            X_temp, y_temp,
                            test_size=(1 - val_ratio),
                            random_state=42
                        )
                else:
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_processed, y,
                        test_size=(test_size + val_size),
                        random_state=42
                    )
                    
                    val_ratio = val_size / (test_size + val_size)
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp,
                        test_size=(1 - val_ratio),
                        random_state=42
                    )
                
                print(f"\nSplit sizes:")
                print(f"  Train: {X_train.shape}")
                print(f"  Val:   {X_val.shape}")
                print(f"  Test:  {X_test.shape}")
                
                # Print class distributions
                print(f"\nTrain distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
                print(f"Val distribution:   {dict(zip(*np.unique(y_val, return_counts=True)))}")
                print(f"Test distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
                
                return X_train, X_val, X_test, y_train, y_val, y_test
                
            except Exception as e:
                print(f"Error during split: {e}")
                raise
        else:
            return X_processed, None, None, None, None, None
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor state"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'scaler': self.scaler,
            'normalization_method': self.normalization_method,
            'input_length': self.input_length
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load saved preprocessor"""
        if not os.path.exists(filepath):
            print(f"No preprocessor found at {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.scaler = state['scaler']
        self.normalization_method = state['normalization_method']
        self.input_length = state['input_length']
        
        print(f"Preprocessor loaded from {filepath}")
        return True
    
    def process_new_data(self, light_curve):
        """
        Process a single new light curve for prediction
        Uses saved preprocessing settings
        """
        # Ensure correct length
        if self.input_length is not None:
            light_curve = self.pad_or_truncate(light_curve, self.input_length)
        
        # Apply preprocessing
        processed = self.preprocess_single_curve(light_curve)
        
        # Reshape for model
        processed = processed.reshape(1, -1, 1)
        
        return processed


# Example usage and testing
if __name__ == "__main__":
    print("Exoplanet Data Preprocessor - Enhanced Version")
    print("="*70)
    
    preprocessor = ExoplanetDataPreprocessor(normalization_method='robust')
    
    # Test with synthetic data
    print("\nTesting with synthetic data...")
    
    # Create synthetic exoplanet signal
    time_points = 3197
    synthetic_lc = np.ones(time_points) + np.random.normal(0, 0.01, time_points)
    
    # Add periodic transits
    period = 400
    transit_duration = 10
    transit_depth = 0.02
    
    for i in range(0, time_points, period):
        synthetic_lc[i:i+transit_duration] -= transit_depth
    
    print(f"Synthetic light curve shape: {synthetic_lc.shape}")
    
    # Process it
    processed = preprocessor.preprocess_single_curve(synthetic_lc)
    print(f"Processed shape: {processed.shape}")
    print(f"Processed mean: {np.mean(processed):.4f}")
    print(f"Processed std: {np.std(processed):.4f}")
    
    print("\nPreprocessor ready for use!")
    print("Use with: python3 hybrid_training.py")