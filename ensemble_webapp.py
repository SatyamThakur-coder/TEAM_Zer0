#!/usr/bin/env python3
"""
ENSEMBLE EXOPLANET DETECTION WEB APPLICATION
Flask web interface using ALL 4 trained models for maximum accuracy
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import os
import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Disable GPU for consistent CPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Global variables for all models
ensemble_models = {
    'hybrid_cnn_lstm': None,
    'multiview_cnn': None, 
    'random_forest': None,
    'som': None
}
model_status = {
    'hybrid_cnn_lstm': False,
    'multiview_cnn': False,
    'random_forest': False,
    'som': False
}
input_length = 3197


def load_ensemble_models():
    """Load all 4 ensemble models"""
    global ensemble_models, model_status
    
    print("\n" + "="*70)
    print("🚀 LOADING ENSEMBLE EXOPLANET DETECTION SYSTEM")
    print("="*70)
    
    models_loaded = 0
    
    # 1. Load Hybrid CNN+LSTM
    print("\n[1/4] Loading Hybrid CNN+LSTM...")
    try:
        model_path = 'models/ensemble_hybrid_cnn_lstm.h5'
        if os.path.exists(model_path):
            ensemble_models['hybrid_cnn_lstm'] = keras.models.load_model(model_path, compile=False)
            model_status['hybrid_cnn_lstm'] = True
            params = ensemble_models['hybrid_cnn_lstm'].count_params()
            print(f"✅ Hybrid CNN+LSTM loaded: {params:,} parameters")
            models_loaded += 1
        else:
            print(f"❌ Model not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading Hybrid CNN+LSTM: {e}")
    
    # 2. Load Multi-View CNN  
    print("\n[2/4] Loading Multi-View CNN...")
    try:
        model_path = 'models/ensemble_multiview_cnn.h5'
        if os.path.exists(model_path):
            ensemble_models['multiview_cnn'] = keras.models.load_model(model_path, compile=False)
            model_status['multiview_cnn'] = True
            params = ensemble_models['multiview_cnn'].count_params()
            print(f"✅ Multi-View CNN loaded: {params:,} parameters")
            models_loaded += 1
        else:
            print(f"❌ Model not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading Multi-View CNN: {e}")
    
    # 3. Load RandomForest
    print("\n[3/4] Loading RandomForest...")
    try:
        model_path = 'models/ensemble_random_forest.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                rf_data = pickle.load(f)
                ensemble_models['random_forest'] = rf_data
                model_status['random_forest'] = True
            print("✅ RandomForest loaded with feature extractor")
            models_loaded += 1
        else:
            print(f"❌ Model not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading RandomForest: {e}")
    
    # 4. Load SOM
    print("\n[4/4] Loading Self-Organizing Map...")
    try:
        model_path = 'models/ensemble_som.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                som_data = pickle.load(f)
                ensemble_models['som'] = som_data
                model_status['som'] = True
            print("✅ SOM loaded with feature extractor")
            models_loaded += 1
        else:
            print(f"❌ Model not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading SOM: {e}")
    
    print("\n" + "="*70)
    print(f"📊 ENSEMBLE STATUS: {models_loaded}/4 models loaded")
    if models_loaded == 4:
        print("🎉 COMPLETE ENSEMBLE - Maximum accuracy available!")
    elif models_loaded >= 2:
        print("⚠️ PARTIAL ENSEMBLE - Reduced accuracy but functional")
    else:
        print("❌ INSUFFICIENT MODELS - Need at least 2 models")
    print("="*70)
    
    return models_loaded >= 2  # Need at least 2 models to work


def normalize_light_curve(light_curve):
    """Normalize using robust method"""
    try:
        light_curve = np.nan_to_num(light_curve, nan=np.nanmedian(light_curve))
        
        median = np.median(light_curve)
        mad = np.median(np.abs(light_curve - median))
        
        if mad > 0:
            light_curve = (light_curve - median) / (1.4826 * mad)
        else:
            light_curve = light_curve - median
        
        light_curve = np.clip(light_curve, -5, 5)
        
        return light_curve
    except Exception as e:
        print(f"Error in normalization: {e}")
        return light_curve


def create_multiview_data(light_curve):
    """Create global and local views for Multi-View CNN"""
    # Global view (2001 points)
    if len(light_curve) > 2001:
        indices = np.linspace(0, len(light_curve)-1, 2001, dtype=int)
        global_view = light_curve[indices]
    elif len(light_curve) < 2001:
        pad_width = 2001 - len(light_curve)
        global_view = np.pad(light_curve, (0, pad_width), mode='constant', 
                           constant_values=np.median(light_curve))
    else:
        global_view = light_curve
    
    # Local view (201 points) - extract around minimum flux point
    transit_center = np.argmin(light_curve)
    half_window = 100  # 201//2
    start = max(0, transit_center - half_window)
    end = min(len(light_curve), transit_center + half_window + 1)
    
    local_view = light_curve[start:end]
    
    if len(local_view) < 201:
        pad_before = (201 - len(local_view)) // 2
        pad_after = 201 - len(local_view) - pad_before
        local_view = np.pad(local_view, (pad_before, pad_after), 
                          mode='constant', constant_values=np.median(light_curve))
    
    local_view = local_view[:201]
    
    return global_view.reshape(1, 2001, 1), local_view.reshape(1, 201, 1)


def make_ensemble_prediction(light_curve):
    """Make prediction using all available models"""
    predictions = {}
    probabilities = {}
    errors = []
    
    # Normalize light curve
    light_curve_norm = normalize_light_curve(light_curve)
    
    # 1. Hybrid CNN+LSTM prediction
    if model_status['hybrid_cnn_lstm']:
        try:
            input_data = light_curve_norm.reshape(1, -1, 1)
            prob = float(ensemble_models['hybrid_cnn_lstm'].predict(input_data, verbose=0)[0][0])
            predictions['Hybrid CNN+LSTM'] = 1 if prob > 0.5 else 0
            probabilities['Hybrid CNN+LSTM'] = prob
        except Exception as e:
            errors.append(f"CNN+LSTM: {str(e)}")
    
    # 2. Multi-View CNN prediction
    if model_status['multiview_cnn']:
        try:
            global_view, local_view = create_multiview_data(light_curve_norm)
            prob = float(ensemble_models['multiview_cnn'].predict([global_view, local_view], verbose=0)[0][0])
            predictions['Multi-View CNN'] = 1 if prob > 0.5 else 0
            probabilities['Multi-View CNN'] = prob
        except Exception as e:
            errors.append(f"Multi-View CNN: {str(e)}")
    
    # 3. RandomForest prediction
    if model_status['random_forest']:
        try:
            rf_data = ensemble_models['random_forest']
            feature_extractor = rf_data['feature_extractor']
            model = rf_data['model']
            
            # Extract features
            features = feature_extractor.extract_features(light_curve_norm).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0)
            
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]
            predictions['RandomForest'] = int(pred)
            probabilities['RandomForest'] = float(prob)
        except Exception as e:
            errors.append(f"RandomForest: {str(e)}")
    
    # 4. SOM prediction
    if model_status['som']:
        try:
            som_data = ensemble_models['som']
            feature_extractor = som_data['feature_extractor']
            som_model = som_data['som_model']
            
            # Extract features
            features = feature_extractor.extract_features(light_curve_norm).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0)
            
            # Get anomaly score
            anomaly_scores, is_anomaly = som_model.predict_anomaly(features, threshold_percentile=85)
            pred = int(is_anomaly[0])
            prob = float((anomaly_scores[0] - np.min(som_data['anomaly_scores'])) / 
                        (np.max(som_data['anomaly_scores']) - np.min(som_data['anomaly_scores'])))
            
            predictions['SOM Anomaly'] = pred
            probabilities['SOM Anomaly'] = prob
        except Exception as e:
            errors.append(f"SOM: {str(e)}")
    
    # Ensemble prediction (weighted voting)
    if probabilities:
        # Simple average for now (could be weighted by model performance)
        ensemble_prob = np.mean(list(probabilities.values()))
        ensemble_pred = 1 if ensemble_prob > 0.5 else 0
        
        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_prob,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'errors': errors,
            'models_used': len(probabilities)
        }
    else:
        return {
            'error': 'No models available for prediction',
            'errors': errors
        }


def plot_to_base64(fig):
    """Convert matplotlib figure to base64"""
    try:
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_data = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        return plot_data
    except Exception as e:
        print(f"Error creating plot: {e}")
        plt.close(fig)
        return None


@app.route('/')
def index():
    """Main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ensemble Exoplanet Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .model-status { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 30px; }
            .model-card { padding: 15px; border-radius: 8px; flex: 1; min-width: 150px; }
            .loaded { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .not-loaded { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 8px; margin-bottom: 20px; }
            .results { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .progress { display: none; text-align: center; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌌 Ensemble Exoplanet Detection System</h1>
                <p>Upload a light curve CSV file to detect exoplanets using 4 AI models</p>
            </div>
            
            <div class="model-status" id="modelStatus">
                <div class="model-card not-loaded">🧠 CNN+LSTM: Not Loaded</div>
                <div class="model-card not-loaded">👁️ Multi-View: Not Loaded</div>
                <div class="model-card not-loaded">🌳 RandomForest: Not Loaded</div>
                <div class="model-card not-loaded">🗺️ SOM: Not Loaded</div>
            </div>
            
            <div class="upload-area">
                <input type="file" id="fileInput" accept=".csv" style="display: none;">
                <button onclick="document.getElementById('fileInput').click()">📁 Select CSV File</button>
                <p>Or drag and drop a CSV file here</p>
            </div>
            
            <div class="progress" id="progress">
                <p>🔄 Analyzing light curve with ensemble models...</p>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            // Check model status on load
            fetch('/model_status')
                .then(response => response.json())
                .then(data => updateModelStatus(data))
                .catch(error => console.error('Error:', error));
            
            function updateModelStatus(status) {
                const statusDiv = document.getElementById('modelStatus');
                statusDiv.innerHTML = `
                    <div class="model-card ${status.hybrid_cnn_lstm ? 'loaded' : 'not-loaded'}">
                        🧠 CNN+LSTM: ${status.hybrid_cnn_lstm ? 'Loaded' : 'Not Loaded'}
                    </div>
                    <div class="model-card ${status.multiview_cnn ? 'loaded' : 'not-loaded'}">
                        👁️ Multi-View: ${status.multiview_cnn ? 'Loaded' : 'Not Loaded'}
                    </div>
                    <div class="model-card ${status.random_forest ? 'loaded' : 'not-loaded'}">
                        🌳 RandomForest: ${status.random_forest ? 'Loaded' : 'Not Loaded'}
                    </div>
                    <div class="model-card ${status.som ? 'loaded' : 'not-loaded'}">
                        🗺️ SOM: ${status.som ? 'Loaded' : 'Not Loaded'}
                    </div>
                `;
            }
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    uploadFile(file);
                }
            });
            
            function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('progress').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('progress').style.display = 'none';
                    displayResults(data);
                })
                .catch(error => {
                    document.getElementById('progress').style.display = 'none';
                    document.getElementById('results').innerHTML = 
                        '<div style="color: red;">Error: ' + error + '</div>';
                });
            }
            
            function displayResults(data) {
                let html = '<div class="results"><h2>🔍 Analysis Results</h2>';
                
                if (data.error) {
                    html += '<div style="color: red;"><strong>Error:</strong> ' + data.error + '</div>';
                } else {
                    const prediction = data.ensemble_prediction === 1 ? 'EXOPLANET DETECTED!' : 'No Exoplanet';
                    const confidence = (Math.abs(data.ensemble_probability - 0.5) * 200).toFixed(1);
                    
                    html += '<div style="text-align: center; margin: 20px 0;">';
                    html += '<h3 style="color: ' + (data.ensemble_prediction === 1 ? '#28a745' : '#6c757d') + ';">';
                    html += '🎯 ' + prediction + '</h3>';
                    html += '<p><strong>Ensemble Probability:</strong> ' + (data.ensemble_probability * 100).toFixed(1) + '%</p>';
                    html += '<p><strong>Confidence:</strong> ' + confidence + '%</p>';
                    html += '<p><strong>Models Used:</strong> ' + data.models_used + '/4</p>';
                    html += '</div>';
                    
                    html += '<h4>📊 Individual Model Results:</h4><table style="width: 100%; border-collapse: collapse;">';
                    html += '<tr><th style="border: 1px solid #ddd; padding: 8px;">Model</th><th style="border: 1px solid #ddd; padding: 8px;">Prediction</th><th style="border: 1px solid #ddd; padding: 8px;">Probability</th></tr>';
                    
                    for (const [model, prob] of Object.entries(data.individual_probabilities)) {
                        const pred = data.individual_predictions[model] === 1 ? 'Exoplanet' : 'No Planet';
                        html += '<tr>';
                        html += '<td style="border: 1px solid #ddd; padding: 8px;">' + model + '</td>';
                        html += '<td style="border: 1px solid #ddd; padding: 8px;">' + pred + '</td>';
                        html += '<td style="border: 1px solid #ddd; padding: 8px;">' + (prob * 100).toFixed(1) + '%</td>';
                        html += '</tr>';
                    }
                    html += '</table>';
                    
                    if (data.plot) {
                        html += '<div style="text-align: center; margin-top: 20px;">';
                        html += '<img src="data:image/png;base64,' + data.plot + '" style="max-width: 100%;" />';
                        html += '</div>';
                    }
                }
                
                html += '</div>';
                document.getElementById('results').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """


@app.route('/model_status')
def get_model_status():
    """Get current model loading status"""
    return jsonify(model_status)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not any(model_status.values()):
            return jsonify({'error': 'No models loaded. Please check model files.'}), 400
        
        light_curve = None
        
        if 'file' in request.files:
            file = request.files['file']
            
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Only CSV files supported'}), 400
            
            try:
                df = pd.read_csv(file)
                
                if len(df) == 1:
                    light_curve = df.iloc[0].values
                else:
                    light_curve = df.iloc[0].values
                
                light_curve = light_curve[~np.isnan(light_curve.astype(float))]
                
            except Exception as e:
                return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No file provided'}), 400
        
        if light_curve is None or len(light_curve) == 0:
            return jsonify({'error': 'No valid data in file'}), 400
        
        # Ensure correct length
        if len(light_curve) != input_length:
            if len(light_curve) < input_length:
                pad_value = np.median(light_curve)
                light_curve = np.pad(light_curve, (0, input_length - len(light_curve)), 
                                   'constant', constant_values=pad_value)
            else:
                light_curve = light_curve[:input_length]
        
        light_curve_original = light_curve.copy()
        
        # Make ensemble prediction
        result = make_ensemble_prediction(light_curve)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Original light curve
        ax1.plot(light_curve_original, linewidth=0.8, color='#667eea', alpha=0.8)
        ax1.set_title('Original Light Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Flux')
        ax1.grid(True, alpha=0.3)
        
        if result['ensemble_prediction'] == 1:
            threshold = np.mean(light_curve_original) - 2 * np.std(light_curve_original)
            dips = light_curve_original < threshold
            ax1.fill_between(range(len(light_curve_original)), 
                           light_curve_original.min(), 
                           light_curve_original.max(),
                           where=dips, alpha=0.3, color='red', 
                           label='Potential Transit')
            ax1.legend()
        
        # Model predictions comparison
        models = list(result['individual_probabilities'].keys())
        probs = [result['individual_probabilities'][m] * 100 for m in models]
        
        bars = ax2.bar(range(len(models)), probs, alpha=0.7)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax2.set_title('Model Predictions Comparison', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Exoplanet Probability (%)')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Color bars based on prediction
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            if prob > 50:
                bar.set_color('#28a745')  # Green for exoplanet
            else:
                bar.set_color('#6c757d')  # Gray for no exoplanet
        
        plt.tight_layout()
        plot_data = plot_to_base64(fig)
        
        if plot_data is None:
            return jsonify({'error': 'Error generating visualization'}), 500
        
        response_data = result.copy()
        response_data['plot'] = plot_data
        response_data['data_points'] = len(light_curve_original)
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict: {error_details}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    models_loaded = sum(model_status.values())
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'total_models': 4,
        'model_status': model_status,
        'ensemble_ready': models_loaded >= 2
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌌 ENSEMBLE EXOPLANET DETECTION WEB APPLICATION")
    print("="*70)
    
    # Load all models
    success = load_ensemble_models()
    
    if not success:
        print("\n❌ ERROR: Not enough models loaded!")
        print("Please ensure model files exist in models/ directory:")
        print("  - models/ensemble_hybrid_cnn_lstm.h5")
        print("  - models/ensemble_multiview_cnn.h5")
        print("  - models/ensemble_random_forest.pkl")
        print("  - models/ensemble_som.pkl")
        sys.exit(1)
    
    print(f"\n🚀 Starting ensemble webapp...")
    print(f"Models loaded: {sum(model_status.values())}/4")
    print(f"Open browser: http://localhost:5000")
    print("="*70)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")