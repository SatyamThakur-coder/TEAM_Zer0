#!/usr/bin/env python3
"""
MODEL DIAGNOSTICS SCRIPT
Analyze your models to detect underfitting, overfitting, or perfect fit
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import learning_curve, validation_curve
from sklearn.model_selection import cross_val_score
import os
import pickle
import tensorflow as tf
from hybrid_cnn_lstm_model import HybridCNNLSTM
from multiview_cnn_model import MultiViewCNN


def analyze_learning_curves(model, X_train, y_train, title="Learning Curve"):
    """
    Analyze learning curves to detect underfitting/overfitting
    """
    print(f"\n🔍 Analyzing {title}...")
    
    # Generate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, 
        cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc'
    )
    
    # Calculate means and stds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('AUC Score')
    plt.title(f'{title} - Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Diagnostic analysis
    plt.subplot(1, 2, 2)
    gap = train_mean - val_mean
    plt.plot(train_sizes, gap, 'o-', color='green', label='Training-Validation Gap')
    plt.axhline(y=0.05, color='orange', linestyle='--', label='Overfitting Threshold (5%)')
    plt.axhline(y=0.02, color='red', linestyle='--', label='Good Fit Threshold (2%)')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score Gap (Train - Val)')
    plt.title('Overfitting Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/diagnostics', exist_ok=True)
    plt.savefig(f'results/diagnostics/{title.lower().replace(" ", "_")}_learning_curve.png', dpi=300)
    plt.close()
    
    # Diagnosis
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    final_gap = gap[-1]
    
    print(f"📊 DIAGNOSIS for {title}:")
    print(f"   Final Training AUC: {final_train:.4f}")
    print(f"   Final Validation AUC: {final_val:.4f}")
    print(f"   Train-Val Gap: {final_gap:.4f}")
    
    if final_gap > 0.05:
        print("   🔴 OVERFITTING DETECTED!")
        print("   Solutions: Reduce model complexity, add regularization, get more data")
    elif final_val < 0.7:
        print("   🟡 UNDERFITTING DETECTED!")
        print("   Solutions: Increase model complexity, train longer, better features")
    elif final_gap < 0.02:
        print("   🟢 GOOD FIT! Model is well-balanced")
    else:
        print("   🟠 SLIGHT OVERFITTING - Monitor closely")
    
    return final_train, final_val, final_gap


def analyze_deep_learning_history(history, model_name="Deep Learning Model"):
    """
    Analyze training history for deep learning models
    """
    print(f"\n🧠 Analyzing {model_name} Training History...")
    
    if history is None:
        print("   ⚠️ No training history available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0,0].plot(history.history['loss'], label='Training Loss', color='blue')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axes[0,0].set_title('Loss Curves')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'accuracy' in history.history:
        axes[0,1].plot(history.history['accuracy'], label='Training Acc', color='blue')
        axes[0,1].plot(history.history['val_accuracy'], label='Validation Acc', color='red')
        axes[0,1].set_title('Accuracy Curves')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # AUC curves
    if 'auc' in history.history:
        axes[1,0].plot(history.history['auc'], label='Training AUC', color='blue')
        axes[1,0].plot(history.history['val_auc'], label='Validation AUC', color='red')
        axes[1,0].set_title('AUC Curves')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('AUC')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1,1].set_title('Diagnostic Summary')
    axes[1,1].axis('off')
    
    # Analyze overfitting
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    loss_gap = abs(final_val_loss - final_train_loss)
    
    # Get best validation loss
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    
    diagnosis_text = f"""
DIAGNOSIS for {model_name}:

Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}
Loss Gap: {loss_gap:.4f}

Best Val Loss: {best_val_loss:.4f}
Best Epoch: {best_epoch}
Total Epochs: {len(history.history['loss'])}

"""
    
    # Determine fit quality
    if loss_gap > 0.3:
        diagnosis_text += "🔴 OVERFITTING!\n"
        diagnosis_text += "• Val loss > Train loss\n"
        diagnosis_text += "• Reduce epochs or add regularization\n"
    elif final_val_loss > 1.0:
        diagnosis_text += "🟡 UNDERFITTING!\n"
        diagnosis_text += "• High validation loss\n"
        diagnosis_text += "• Train longer or increase complexity\n"
    elif best_epoch < len(history.history['loss']) * 0.7:
        diagnosis_text += "🟠 EARLY CONVERGENCE\n"
        diagnosis_text += "• Model converged early\n"
        diagnosis_text += "• Could benefit from longer training\n"
    else:
        diagnosis_text += "🟢 GOOD FIT!\n"
        diagnosis_text += "• Well-balanced training\n"
        diagnosis_text += "• Good convergence pattern\n"
    
    axes[1,1].text(0.05, 0.95, diagnosis_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'results/diagnostics/{model_name.lower().replace(" ", "_")}_history.png', dpi=300)
    plt.close()
    
    print(f"📊 {model_name} Analysis Complete!")
    print(f"   Final Train Loss: {final_train_loss:.4f}")
    print(f"   Final Val Loss: {final_val_loss:.4f}")
    print(f"   Best Epoch: {best_epoch}/{len(history.history['loss'])}")


def check_model_complexity():
    """
    Check if model complexity matches data complexity
    """
    print("\n🔧 Model Complexity Analysis...")
    
    models_info = {}
    
    # Check saved models
    models_dir = 'models'
    if os.path.exists(models_dir):
        for model_file in os.listdir(models_dir):
            file_path = os.path.join(models_dir, model_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            if model_file.endswith('.h5'):
                # TensorFlow model
                try:
                    model = tf.keras.models.load_model(file_path)
                    params = model.count_params()
                    models_info[model_file] = {
                        'type': 'Deep Learning',
                        'parameters': params,
                        'size_mb': file_size,
                        'complexity': 'High' if params > 1000000 else 'Medium' if params > 100000 else 'Low'
                    }
                except:
                    models_info[model_file] = {
                        'type': 'Deep Learning',
                        'parameters': 'Unknown',
                        'size_mb': file_size,
                        'complexity': 'Unknown'
                    }
            
            elif model_file.endswith('.pkl'):
                models_info[model_file] = {
                    'type': 'Traditional ML',
                    'parameters': 'Variable',
                    'size_mb': file_size,
                    'complexity': 'Medium'
                }
    
    # Display results
    print("\n📊 MODEL COMPLEXITY REPORT:")
    print("="*70)
    for model_name, info in models_info.items():
        print(f"{model_name}:")
        print(f"  Type: {info['type']}")
        print(f"  Parameters: {info['parameters']:,}" if isinstance(info['parameters'], int) else f"  Parameters: {info['parameters']}")
        print(f"  Size: {info['size_mb']:.1f} MB")
        print(f"  Complexity: {info['complexity']}")
        print()
    
    # Recommendations
    total_size = sum(info['size_mb'] for info in models_info.values())
    print(f"Total Model Size: {total_size:.1f} MB")
    
    if total_size > 200:
        print("⚠️  Large model ensemble - consider model compression")
    elif total_size < 10:
        print("✅ Compact model ensemble - good for deployment")
    else:
        print("✅ Reasonable model size")


def generate_diagnostic_report():
    """
    Generate comprehensive diagnostic report
    """
    print("\n" + "="*70)
    print("🏥 COMPREHENSIVE MODEL DIAGNOSTICS")
    print("="*70)
    
    os.makedirs('results/diagnostics', exist_ok=True)
    
    # Check model complexity
    check_model_complexity()
    
    # Check if we have training histories available
    models_to_check = [
        ('models/ensemble_hybrid_cnn_lstm.h5', 'Hybrid CNN+LSTM'),
        ('models/ensemble_multiview_cnn.h5', 'Multi-View CNN'),
        ('models/ensemble_random_forest.pkl', 'RandomForest'),
        ('models/ensemble_som.pkl', 'SOM')
    ]
    
    available_models = []
    for model_path, model_name in models_to_check:
        if os.path.exists(model_path):
            available_models.append((model_path, model_name))
            print(f"✅ {model_name}: Available")
        else:
            print(f"❌ {model_name}: Not found")
    
    # Create diagnostic summary
    with open('results/diagnostics/diagnostic_report.txt', 'w') as f:
        f.write("EXOPLANET DETECTION MODEL DIAGNOSTICS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("HOW TO INTERPRET RESULTS:\n\n")
        
        f.write("🟢 GOOD FIT Signs:\n")
        f.write("- Training and validation curves converge\n")
        f.write("- Small gap between train/val performance (<2%)\n")
        f.write("- Validation loss decreases steadily\n")
        f.write("- Model generalizes well to test data\n\n")
        
        f.write("🔴 OVERFITTING Signs:\n")
        f.write("- Training accuracy >> Validation accuracy\n")
        f.write("- Training loss << Validation loss\n")
        f.write("- Large gap between train/val performance (>5%)\n")
        f.write("- Validation loss increases while training loss decreases\n")
        f.write("Solutions: Reduce model complexity, add regularization, more data\n\n")
        
        f.write("🟡 UNDERFITTING Signs:\n")
        f.write("- Both training and validation accuracy are low\n")
        f.write("- Both losses remain high\n")
        f.write("- Performance plateaus early\n")
        f.write("- Model too simple for the data\n")
        f.write("Solutions: Increase model complexity, train longer, better features\n\n")
        
        f.write("📊 YOUR MODELS:\n")
        for model_path, model_name in available_models:
            f.write(f"- {model_name}: {model_path}\n")
        
        f.write(f"\nTotal Models: {len(available_models)}\n")
        
        if len(available_models) == 4:
            f.write("✅ Complete ensemble - all models available!\n")
        else:
            f.write(f"⚠️ Incomplete ensemble - {4-len(available_models)} models missing\n")
    
    print(f"\n✅ Diagnostic report saved to: results/diagnostics/diagnostic_report.txt")
    print("\n🎯 QUICK DIAGNOSTICS GUIDE:")
    print("""
    🟢 GOOD FIT: Train/Val curves close together, both improving
    🔴 OVERFITTING: Train performance >> Val performance  
    🟡 UNDERFITTING: Both train/val performance low
    🟠 EARLY STOPPING: Training stopped too early, could improve more
    
    Check the generated visualizations in results/diagnostics/
    """)


if __name__ == "__main__":
    generate_diagnostic_report()