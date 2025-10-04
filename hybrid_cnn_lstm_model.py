"""
HYBRID CNN+LSTM MODEL FOR EXOPLANET DETECTION
Combines CNN feature extraction with LSTM temporal modeling
Best for detecting periodic patterns in light curves
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt


class HybridCNNLSTM:
    """
    Hybrid CNN+LSTM architecture for exoplanet detection
    
    Architecture:
    1. CNN layers extract local features (transit shapes)
    2. LSTM layers model temporal dependencies (periodicity)
    3. Dense layers for final classification
    
    Advantages over pure CNN:
    - Better at detecting periodic transits
    - Captures long-range temporal dependencies
    - More robust to noise in time series
    """
    
    def __init__(self, input_shape=(3197, 1), model_path='models/exoplanet_cnn_lstm.h5'):
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Build Hybrid CNN+LSTM architecture
        
        Flow:
        Input → CNN Feature Extraction → LSTM Temporal Modeling → Dense Classification
        """
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # ==================== CNN FEATURE EXTRACTION ====================
        # Extract local patterns (individual transit shapes)
        
        # CNN Block 1: Detect short-term features
        x = layers.Conv1D(32, kernel_size=5, activation='relu', 
                         padding='same', name='conv1d_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.MaxPooling1D(pool_size=5, name='pool_1')(x)
        x = layers.Dropout(0.2, name='dropout_1')(x)
        
        # CNN Block 2: Detect medium-term features
        x = layers.Conv1D(64, kernel_size=5, activation='relu', 
                         padding='same', name='conv1d_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.MaxPooling1D(pool_size=5, name='pool_2')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        
        # CNN Block 3: Detect long-term features
        x = layers.Conv1D(128, kernel_size=5, activation='relu', 
                         padding='same', name='conv1d_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.MaxPooling1D(pool_size=5, name='pool_3')(x)
        x = layers.Dropout(0.3, name='dropout_3')(x)
        
        # ==================== LSTM TEMPORAL MODELING ====================
        # Model sequential dependencies (orbital periodicity)
        
        # Bidirectional LSTM 1: Learn forward and backward patterns
        # MAXIMUM PERFORMANCE - Optimized for best accuracy
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, 
                       dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_1'
        )(x)
        x = layers.BatchNormalization(name='bn_lstm_1')(x)
        
        # Bidirectional LSTM 2: Higher-level temporal features
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=False,
                       dropout=0.3, recurrent_dropout=0.2),
            name='bilstm_2'
        )(x)
        x = layers.BatchNormalization(name='bn_lstm_2')(x)
        
        # ==================== DENSE CLASSIFICATION ====================
        # Combine features for final decision
        
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='dense_1')(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        x = layers.Dropout(0.4, name='dropout_dense_1')(x)
        
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='dense_2')(x)
        x = layers.Dropout(0.4, name='dropout_dense_2')(x)
        
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='dense_3')(x)
        x = layers.Dropout(0.3, name='dropout_dense_3')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, 
                                  name='Hybrid_CNN_LSTM')
        
        print("✓ Hybrid CNN+LSTM architecture created!")
        print("  - CNN layers: Extract local transit features")
        print("  - LSTM layers: Model temporal periodicity")
        print("  - Dense layers: Final classification")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with appropriate loss and metrics"""
        if self.model is None:
            self.build_model()
        
        # Use Adam optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Binary crossentropy for exoplanet detection
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.FalseNegatives(name='fn'),
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.TrueNegatives(name='tn')
            ]
        )
        print("✓ Model compiled successfully!")
    
    def lr_schedule(self, epoch, lr):
        """Learning rate scheduler - decay over time"""
        if epoch < 10:
            return lr
        elif epoch < 30:
            return lr * 0.5
        else:
            return lr * 0.1
    
    def get_callbacks(self):
        """Enhanced training callbacks for hybrid model"""
        callbacks = [
            # Stop if validation loss doesn't improve
            EarlyStopping(
                monitor='val_auc',
                patience=20,  # More patience for LSTM
                restore_best_weights=True,
                verbose=1,
                mode='max',
                min_delta=0.001
            ),
            
            # Save best model
            ModelCheckpoint(
                self.model_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                min_delta=0.0001
            ),
            
            # Learning rate scheduler
            LearningRateScheduler(self.lr_schedule, verbose=0),
            
            # Progress callback
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print(
                    f"Epoch {epoch + 1}: "
                    f"Loss: {logs['loss']:.4f} - "
                    f"Acc: {logs['accuracy']:.4f} - "
                    f"AUC: {logs['auc']:.4f} - "
                    f"Val_Loss: {logs['val_loss']:.4f} - "
                    f"Val_Acc: {logs['val_accuracy']:.4f} - "
                    f"Val_AUC: {logs['val_auc']:.4f}"
                ) if (epoch + 1) % 5 == 0 else None
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, 
              batch_size=32, class_weight=None, initial_epoch=0, checkpoint_dir=None):
        """
        Train the hybrid model
        
        Args:
            initial_epoch: Starting epoch (for resuming training)
            checkpoint_dir: Directory to save checkpoints (enables resume capability)
        
        Note: LSTM requires more training time than pure CNN
        """
        if self.model is None:
            self.compile_model()
        
        # Calculate class weights if not provided
        if class_weight is None and y_train is not None:
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            total = neg + pos
            
            class_weight = {
                0: (1 / neg) * (total / 2.0),
                1: (1 / pos) * (total / 2.0)
            }
            
            print(f"\nClass weights calculated:")
            print(f"  Class 0 (No Planet): {class_weight[0]:.3f}")
            print(f"  Class 1 (Exoplanet): {class_weight[1]:.3f}\n")
        
        print("Starting training (Hybrid CNN+LSTM)...")
        print("Note: LSTM training is slower but captures better temporal patterns")
        print("=" * 70)
        
        # Get callbacks and add checkpoint callback if enabled
        callbacks = self.get_callbacks()
        
        if checkpoint_dir:
            # Add custom checkpoint callback to save after EVERY epoch
            checkpoint_callback = keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._save_checkpoint(
                    epoch, checkpoint_dir
                )
            )
            callbacks.append(checkpoint_callback)
            print(f"Checkpoint saving enabled: {checkpoint_dir}")
            print("Model will auto-save after every epoch")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=initial_epoch
        )
        
        print("=" * 70)
        print("✓ Training completed!\n")
        return self.history
    
    def _save_checkpoint(self, epoch, checkpoint_dir):
        """Save checkpoint after each epoch for resume capability"""
        import os
        
        # Save model
        checkpoint_model_path = os.path.join(checkpoint_dir, 'model_checkpoint.keras')
        self.model.save(checkpoint_model_path)
        
        # Save epoch number
        checkpoint_epoch_file = os.path.join(checkpoint_dir, 'last_epoch.txt')
        with open(checkpoint_epoch_file, 'w') as f:
            f.write(str(epoch + 1))  # +1 because epoch is 0-indexed
        
        # Save history
        if self.history is not None:
            history_file = os.path.join(checkpoint_dir, 'history.npz')
            np.savez_compressed(
                history_file,
                history=self.history.history
            )
        
        # Only print every 5 epochs to reduce clutter
        if (epoch + 1) % 5 == 0:
            print(f"✓ Checkpoint saved at epoch {epoch + 1}")
    
    def evaluate(self, X_test, y_test):
        """Enhanced evaluation with confusion matrix components"""
        print("Evaluating Hybrid CNN+LSTM model on test set...")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        metrics = dict(zip(self.model.metrics_names, results))

        # Calculate additional metrics - handle missing keys gracefully
        try:
            tp = int(metrics.get('tp', 0))
            tn = int(metrics.get('tn', 0))
            fp = int(metrics.get('fp', 0))
            fn = int(metrics.get('fn', 0))
        except (KeyError, ValueError):
            # If metrics are not available, calculate manually
            print("  Calculating confusion matrix metrics manually...")
            predictions = self.model.predict(X_test, verbose=0)
            predictions = (predictions > 0.5).astype(int).flatten()

            tp = int(np.sum((predictions == 1) & (y_test == 1)))
            tn = int(np.sum((predictions == 0) & (y_test == 0)))
            fp = int(np.sum((predictions == 1) & (y_test == 0)))
            fn = int(np.sum((predictions == 0) & (y_test == 1)))

            # Update metrics with calculated values
            metrics['tp'] = tp
            metrics['tn'] = tn
            metrics['fp'] = fp
            metrics['fn'] = fn
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        print("\n" + "="*70)
        print("TEST SET RESULTS - HYBRID CNN+LSTM MODEL")
        print("="*70)
        print(f"Loss:             {metrics['loss']:.4f}")
        # Handle different metric name formats
        accuracy_key = 'accuracy' if 'accuracy' in metrics else 'acc' if 'acc' in metrics else None
        precision_key = 'precision' if 'precision' in metrics else 'prec' if 'prec' in metrics else None
        recall_key = 'recall' if 'recall' in metrics else 'rec' if 'rec' in metrics else None
        
        if accuracy_key:
            print(f"Accuracy:         {metrics[accuracy_key]:.4f} ({metrics[accuracy_key]*100:.2f}%)")
        if precision_key:
            print(f"Precision:        {metrics[precision_key]:.4f} ({metrics[precision_key]*100:.2f}%)")
        if recall_key:
            print(f"Recall:           {metrics[recall_key]:.4f} ({metrics[recall_key]*100:.2f}%)")
        print(f"Specificity:      {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"F1 Score:         {f1_score:.4f} ({f1_score*100:.2f}%)")
        auc_key = 'auc' if 'auc' in metrics else 'auc_roc' if 'auc_roc' in metrics else None
        if auc_key:
            print(f"AUC-ROC:          {metrics[auc_key]:.4f}")
        
        print("-"*70)
        print(f"True Positives:   {tp}")
        print(f"True Negatives:   {tn}")
        print(f"False Positives:  {fp}")
        print(f"False Negatives:  {fn}")
        print("="*70 + "\n")
        
        # Ensure all required keys exist for ensemble
        result_metrics = {
            'loss': metrics.get('loss', 0.0),
            'accuracy': metrics.get(accuracy_key, 0.0) if accuracy_key else 0.0,
            'precision': metrics.get(precision_key, 0.0) if precision_key else 0.0,
            'recall': metrics.get(recall_key, 0.0) if recall_key else 0.0,
            'auc': metrics.get(auc_key, 0.0) if auc_key else 0.0,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        return result_metrics
    
    def predict(self, X, threshold=0.5):
        """Make predictions with the hybrid model"""
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities > threshold).astype(int)
        
        n_exoplanets = np.sum(predictions == 1)
        print(f"Predictions: {n_exoplanets} exoplanets detected out of {len(X)} samples")
        print(f"Detection rate: {n_exoplanets/len(X)*100:.2f}%")
        
        return predictions, probabilities
    
    def plot_training_history(self, save_path='training_history_cnn_lstm.png'):
        """Enhanced training history visualization for hybrid model"""
        if self.history is None:
            print("No training history available")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history.history['loss'], label='Train Loss', linewidth=2, color='#e74c3c')
        ax1.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2, color='#3498db')
        ax1.set_title('Model Loss (CNN+LSTM)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.history.history['accuracy'], label='Train Acc', linewidth=2, color='#e74c3c')
        ax2.plot(self.history.history['val_accuracy'], label='Val Acc', linewidth=2, color='#3498db')
        ax2.set_title('Model Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # AUC
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.history.history['auc'], label='Train AUC', linewidth=2, color='#e74c3c')
        ax3.plot(self.history.history['val_auc'], label='Val AUC', linewidth=2, color='#3498db')
        ax3.set_title('AUC-ROC Score', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AUC')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Precision & Recall
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.history.history['precision'], label='Train Precision', linewidth=2)
        ax4.plot(self.history.history['recall'], label='Train Recall', linewidth=2)
        ax4.plot(self.history.history['val_precision'], label='Val Precision', linewidth=2, linestyle='--')
        ax4.plot(self.history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
        ax4.set_title('Precision & Recall', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # False Positives & False Negatives
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.history.history['fp'], label='False Positives', linewidth=2, color='red')
        ax5.plot(self.history.history['fn'], label='False Negatives', linewidth=2, color='orange')
        ax5.set_title('False Predictions', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Count')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # True Positives & True Negatives
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.history.history['tp'], label='True Positives', linewidth=2, color='green')
        ax6.plot(self.history.history['tn'], label='True Negatives', linewidth=2, color='blue')
        ax6.set_title('True Predictions', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Count')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Learning Rate
        ax7 = fig.add_subplot(gs[2, 0])
        if 'lr' in self.history.history:
            ax7.plot(self.history.history['lr'], linewidth=2, color='purple')
            ax7.set_yscale('log')
        ax7.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Learning Rate')
        ax7.grid(True, alpha=0.3)
        
        # Training Summary
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        best_epoch = np.argmax(self.history.history['val_auc']) + 1
        summary_text = "="*50 + "\n"
        summary_text += "HYBRID CNN+LSTM TRAINING SUMMARY\n"
        summary_text += "="*50 + "\n\n"
        summary_text += f"Total Epochs:        {len(self.history.history['loss'])}\n"
        summary_text += f"Best Epoch:          {best_epoch}\n\n"
        summary_text += f"Best Val Loss:       {min(self.history.history['val_loss']):.4f}\n"
        summary_text += f"Best Val Accuracy:   {max(self.history.history['val_accuracy']):.4f}\n"
        summary_text += f"Best Val AUC:        {max(self.history.history['val_auc']):.4f}\n"
        summary_text += f"Best Val Precision:  {max(self.history.history['val_precision']):.4f}\n"
        summary_text += f"Best Val Recall:     {max(self.history.history['val_recall']):.4f}\n\n"
        summary_text += "Architecture:\n"
        summary_text += "  • CNN: 3 blocks (32→64→128 filters)\n"
        summary_text += "  • LSTM: 2 bidirectional layers (128→64 units)\n"
        summary_text += "  • Dense: 3 layers (256→128→64→1)\n"
        
        ax8.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Hybrid CNN+LSTM Training History', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
        plt.close()
    
    def load_model(self):
        """Load a saved model"""
        self.model = keras.models.load_model(self.model_path)
        print(f"✓ Hybrid CNN+LSTM model loaded from {self.model_path}")
    
    def summary(self):
        """Print detailed model summary"""
        if self.model is None:
            self.build_model()
        
        print("\n" + "="*70)
        print("HYBRID CNN+LSTM ARCHITECTURE SUMMARY")
        print("="*70)
        self.model.summary()
        print("="*70 + "\n")
        
        # Calculate model size
        total_params = self.model.count_params()
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated model size: {total_params * 4 / (1024**2):.2f} MB")
        print(f"\nModel advantages:")
        print("  • CNN extracts local transit patterns")
        print("  • LSTM captures long-range temporal dependencies")
        print("  • Bidirectional processing for better context")
        print("  • Superior for periodic signal detection")
        print()


# Example usage
if __name__ == "__main__":
    print("Hybrid CNN+LSTM Exoplanet Detection Model")
    print("="*70)
    
    # Create model
    model = HybridCNNLSTM(input_shape=(3197, 1))
    model.build_model()
    model.summary()
    
    print("\nModel ready for training!")
    print("Use exoplanet_training_hybrid.py to train the model.")
    print("\nNote: Training will be slower than pure CNN due to LSTM layers,")
    print("      but will achieve better accuracy on periodic patterns!")