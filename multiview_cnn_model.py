"""
MULTI-VIEW CNN MODEL FOR EXOPLANET DETECTION
Based on Shallue & Vanderburg (2018) - Google Brain's approach
Uses both GLOBAL and LOCAL views of light curves for superior accuracy

This architecture achieved 96% accuracy in the original paper
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt


class MultiViewCNN:
    """
    Multi-View CNN Architecture for Exoplanet Detection
    
    Key Innovation: Process BOTH global and local views simultaneously
    - Global view: Entire light curve (captures overall pattern)
    - Local view: Zoomed transit region (captures fine details)
    
    This dual-input approach significantly outperforms single-view models!
    """
    
    def __init__(self, 
                 global_shape=(2001, 1), 
                 local_shape=(201, 1),
                 model_path='models/multiview_cnn.h5'):
        """
        Args:
            global_shape: Shape of global view (full light curve)
            local_shape: Shape of local view (zoomed transit region)
        """
        self.global_shape = global_shape
        self.local_shape = local_shape
        self.model_path = model_path
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Build Multi-View CNN Architecture
        
        Architecture follows Shallue & Vanderburg (2018):
        - Global CNN tower: 5 conv blocks (16->32->64->128->256)
        - Local CNN tower: 2 conv blocks (16->32)
        - Merged fully connected layers: 512->512->512->1
        """
        
        # ==================== GLOBAL VIEW INPUT ====================
        # Processes entire light curve to capture overall patterns
        input_global = layers.Input(shape=self.global_shape, name='global_input')
        
        # Global Conv Block 1
        x = layers.Conv1D(16, 5, strides=1, activation='relu', padding='same')(input_global)
        x = layers.Conv1D(16, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=5, strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Global Conv Block 2
        x = layers.Conv1D(32, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.Conv1D(32, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=5, strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Global Conv Block 3
        x = layers.Conv1D(64, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.Conv1D(64, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=5, strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Global Conv Block 4
        x = layers.Conv1D(128, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.Conv1D(128, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=5, strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Global Conv Block 5
        x = layers.Conv1D(256, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.Conv1D(256, 5, strides=1, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=5, strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # ==================== LOCAL VIEW INPUT ====================
        # Processes zoomed transit region for fine detail detection
        input_local = layers.Input(shape=self.local_shape, name='local_input')
        
        # Local Conv Block 1
        y = layers.Conv1D(16, 5, strides=1, activation='relu', padding='same')(input_local)
        y = layers.Conv1D(16, 5, strides=1, activation='relu', padding='same')(y)
        y = layers.MaxPooling1D(pool_size=7, strides=2)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.1)(y)
        
        # Local Conv Block 2
        y = layers.Conv1D(32, 5, strides=1, activation='relu', padding='same')(y)
        y = layers.Conv1D(32, 5, strides=1, activation='relu', padding='same')(y)
        y = layers.MaxPooling1D(pool_size=7, strides=2)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)
        
        # ==================== MERGE AND CLASSIFY ====================
        # Combine features from both views
        xf = layers.Flatten()(x)
        yf = layers.Flatten()(y)
        z = layers.Concatenate()([xf, yf])
        
        # Dense classification layers
        z = layers.Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001))(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.4)(z)
        
        z = layers.Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001))(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dropout(0.4)(z)
        
        z = layers.Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001))(z)
        z = layers.Dropout(0.3)(z)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(z)
        
        # Create model with dual inputs
        self.model = models.Model(
            inputs=[input_global, input_local], 
            outputs=output,
            name='MultiView_CNN'
        )
        
        print("✓ Multi-View CNN architecture created!")
        print("  - Global view: Full light curve analysis")
        print("  - Local view: Transit detail analysis")
        print("  - Architecture: Shallue & Vanderburg (2018)")
        print("  - Expected accuracy: ~96% (Google Brain benchmark)")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile with appropriate optimizer and metrics"""
        if self.model is None:
            self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        print("✓ Model compiled successfully!")
    
    def get_callbacks(self):
        """Training callbacks"""
        return [
            EarlyStopping(
                monitor='val_auc',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def train(self, X_train_global, X_train_local, y_train,
              X_val_global, X_val_local, y_val,
              epochs=100, batch_size=32, class_weight=None):
        """
        Train with dual inputs
        
        Args:
            X_train_global: Global view training data
            X_train_local: Local view training data
            y_train: Training labels
            X_val_global: Global view validation data
            X_val_local: Local view validation data
            y_val: Validation labels
        """
        if self.model is None:
            self.compile_model()
        
        # Calculate class weights
        if class_weight is None:
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            total = neg + pos
            class_weight = {
                0: (1 / neg) * (total / 2.0),
                1: (1 / pos) * (total / 2.0)
            }
            print(f"\nClass weights: 0={class_weight[0]:.3f}, 1={class_weight[1]:.3f}")
        
        print("\nStarting Multi-View CNN training...")
        print("=" * 70)
        
        self.history = self.model.fit(
            [X_train_global, X_train_local], y_train,
            validation_data=([X_val_global, X_val_local], y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print("=" * 70)
        print("✓ Training completed!\n")
        return self.history
    
    def evaluate(self, X_test_global, X_test_local, y_test):
        """Evaluate on test set"""
        print("Evaluating Multi-View CNN...")
        results = self.model.evaluate([X_test_global, X_test_local], y_test, verbose=1)
        metrics = dict(zip(self.model.metrics_names, results))
        
        print("\n" + "=" * 70)
        print("MULTI-VIEW CNN TEST RESULTS")
        print("=" * 70)
        # Handle different metric name formats
        accuracy_key = 'accuracy' if 'accuracy' in metrics else 'acc' if 'acc' in metrics else None
        precision_key = 'precision' if 'precision' in metrics else 'prec' if 'prec' in metrics else None
        recall_key = 'recall' if 'recall' in metrics else 'rec' if 'rec' in metrics else None
        auc_key = 'auc' if 'auc' in metrics else 'auc_roc' if 'auc_roc' in metrics else None
        
        if accuracy_key:
            print(f"Accuracy:  {metrics[accuracy_key]:.4f} ({metrics[accuracy_key]*100:.2f}%)")
        if precision_key:
            print(f"Precision: {metrics[precision_key]:.4f}")
        if recall_key:
            print(f"Recall:    {metrics[recall_key]:.4f}")
        if auc_key:
            print(f"AUC-ROC:   {metrics[auc_key]:.4f}")
        print("=" * 70 + "\n")
        
        # Ensure all required keys exist for ensemble
        result_metrics = {
            'loss': metrics.get('loss', 0.0),
            'accuracy': metrics.get(accuracy_key, 0.0) if accuracy_key else 0.0,
            'precision': metrics.get(precision_key, 0.0) if precision_key else 0.0,
            'recall': metrics.get(recall_key, 0.0) if recall_key else 0.0,
            'auc': metrics.get(auc_key, 0.0) if auc_key else 0.0
        }
        
        return result_metrics
    
    def predict(self, X_global, X_local, threshold=0.5):
        """Make predictions"""
        probabilities = self.model.predict([X_global, X_local], verbose=0)
        predictions = (probabilities > threshold).astype(int)
        
        n_exoplanets = np.sum(predictions == 1)
        print(f"Detected: {n_exoplanets} exoplanets out of {len(predictions)} samples")
        return predictions, probabilities
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        
        print("\n" + "=" * 70)
        print("MULTI-VIEW CNN ARCHITECTURE SUMMARY")
        print("=" * 70)
        self.model.summary()
        print("=" * 70)
        
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Estimated size: {total_params * 4 / (1024**2):.2f} MB")
        print("\nReference: Shallue & Vanderburg (2018)")
        print("Original paper accuracy: ~96%\n")
    
    def load_model(self):
        """Load saved model"""
        self.model = keras.models.load_model(self.model_path)
        print(f"✓ Multi-View CNN loaded from {self.model_path}")


def create_local_view(light_curve, transit_center=None, window_size=201):
    """
    Extract local view (zoomed transit region) from light curve
    
    Args:
        light_curve: Full light curve
        transit_center: Center of transit (if None, use minimum flux point)
        window_size: Size of local view window
    
    Returns:
        Local view array
    """
    if transit_center is None:
        # Find potential transit center (minimum flux)
        transit_center = np.argmin(light_curve)
    
    half_window = window_size // 2
    start = max(0, transit_center - half_window)
    end = min(len(light_curve), transit_center + half_window + 1)
    
    local_view = light_curve[start:end]
    
    # Pad if necessary
    if len(local_view) < window_size:
        pad_before = (window_size - len(local_view)) // 2
        pad_after = window_size - len(local_view) - pad_before
        local_view = np.pad(local_view, (pad_before, pad_after), 
                          mode='constant', constant_values=np.median(light_curve))
    
    return local_view[:window_size]


if __name__ == "__main__":
    print("Multi-View CNN for Exoplanet Detection")
    print("=" * 70)
    print("Based on Shallue & Vanderburg (2018)")
    print("Google Brain's state-of-the-art architecture\n")
    
    model = MultiViewCNN()
    model.build_model()
    model.summary()
    
    print("\nAdvantages:")
    print("  • Processes BOTH global and local views")
    print("  • Global view: Captures overall pattern")
    print("  • Local view: Captures fine transit details")
    print("  • 96% accuracy in original paper")
    print("  • Superior to single-view approaches")
    print("\nUse with: python3 ensemble_training.py")
