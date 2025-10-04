"""
HYBRID CNN+LSTM WEB APPLICATION
Flask web interface for exoplanet detection using Hybrid model
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from hybrid_preprocessing import ExoplanetDataPreprocessor

app = Flask(__name__)

# Global variables
model = None
preprocessor = None
input_length = 3197
MODEL_TYPE = "Hybrid CNN+LSTM"

def load_model(model_path='models/exoplanet_cnn_lstm.h5'):
    """Load the trained hybrid model"""
    global model
    try:
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Train the model first: python3 hybrid_training_complete.py")
            return False
        
        model = keras.models.load_model(model_path)
        print(f"Hybrid CNN+LSTM model loaded from {model_path}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def load_preprocessor():
    """Load the preprocessor"""
    global preprocessor
    try:
        preprocessor = ExoplanetDataPreprocessor()
        if os.path.exists('models/preprocessor_hybrid.pkl'):
            preprocessor.load_preprocessor('models/preprocessor_hybrid.pkl')
            print("Preprocessor loaded")
        else:
            print("Using default preprocessor")
        return True
    except Exception as e:
        print(f"Warning: Could not load preprocessor: {e}")
        return False

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
    try:
        return render_template('index_enhanced.html')
    except Exception as e:
        return f"""
        <html>
        <body>
        <h1>Error Loading Interface</h1>
        <p>Error: {str(e)}</p>
        <p>Make sure templates/index_enhanced.html exists</p>
        </body>
        </html>
        """, 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Train first: python3 hybrid_training_complete.py'
            }), 400
        
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
                    print(f"CSV has {len(df)} rows, using first row")
                    light_curve = df.iloc[0].values
                
                light_curve = light_curve[~np.isnan(light_curve.astype(float))]
                
            except Exception as e:
                return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
                
        elif request.json and 'data' in request.json:
            try:
                light_curve = np.array(request.json['data'])
            except Exception as e:
                return jsonify({'error': f'Error parsing data: {str(e)}'}), 400
        else:
            return jsonify({'error': 'No data provided'}), 400
        
        if light_curve is None or len(light_curve) == 0:
            return jsonify({'error': 'No valid data in file'}), 400
        
        print(f"Received light curve: {len(light_curve)} points")
        
        # Ensure correct length
        if len(light_curve) != input_length:
            if len(light_curve) < input_length:
                pad_value = np.median(light_curve)
                light_curve = np.pad(light_curve, (0, input_length - len(light_curve)), 
                                    'constant', constant_values=pad_value)
                print(f"Padded to {input_length}")
            else:
                light_curve = light_curve[:input_length]
                print(f"Truncated to {input_length}")
        
        light_curve_original = light_curve.copy()
        
        # Normalize
        light_curve_normalized = normalize_light_curve(light_curve)
        light_curve_input = light_curve_normalized.reshape(1, -1, 1)
        
        # Predict
        print("Making prediction with Hybrid CNN+LSTM...")
        probability = float(model.predict(light_curve_input, verbose=0)[0][0])
        prediction = "Exoplanet Detected!" if probability > 0.5 else "No Exoplanet"
        
        print(f"Prediction: {prediction} (probability: {probability:.4f})")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        # Original
        ax1.plot(light_curve_original, linewidth=0.8, color='#667eea', alpha=0.8)
        ax1.set_title('Original Light Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Flux')
        ax1.grid(True, alpha=0.3)
        
        if probability > 0.5:
            threshold = np.mean(light_curve_original) - 2 * np.std(light_curve_original)
            dips = light_curve_original < threshold
            ax1.fill_between(range(len(light_curve_original)), 
                           light_curve_original.min(), 
                           light_curve_original.max(),
                           where=dips, alpha=0.3, color='red', 
                           label='Potential Transit')
            ax1.legend()
        
        # Normalized
        ax2.plot(light_curve_normalized, linewidth=0.8, color='#38ef7d', alpha=0.8)
        ax2.set_title('Normalized Light Curve (Model Input)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Normalized Flux')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plot_data = plot_to_base64(fig)
        
        if plot_data is None:
            return jsonify({'error': 'Error generating visualization'}), 500
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'confidence': abs(probability - 0.5) * 2,
            'plot': plot_data,
            'data_points': len(light_curve_original),
            'model_type': MODEL_TYPE
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict: {error_details}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        print(f"Processing batch: {len(df)} samples")
        
        results = []
        for idx, row in df.iterrows():
            try:
                light_curve = row.values
                light_curve = light_curve[~np.isnan(light_curve.astype(float))]
                
                if len(light_curve) != input_length:
                    if len(light_curve) < input_length:
                        pad_value = np.median(light_curve)
                        light_curve = np.pad(light_curve, (0, input_length - len(light_curve)), 
                                           'constant', constant_values=pad_value)
                    else:
                        light_curve = light_curve[:input_length]
                
                light_curve_normalized = normalize_light_curve(light_curve)
                light_curve_input = light_curve_normalized.reshape(1, -1, 1)
                probability = float(model.predict(light_curve_input, verbose=0)[0][0])
                
                results.append({
                    'sample_id': idx + 1,
                    'probability': probability,
                    'prediction': 'Exoplanet' if probability > 0.5 else 'No Exoplanet'
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append({
                    'sample_id': idx + 1,
                    'probability': -1,
                    'prediction': 'Error'
                })
        
        exoplanet_count = sum(1 for r in results if r['prediction'] == 'Exoplanet')
        
        return jsonify({
            'results': results, 
            'total': len(results),
            'exoplanets_detected': exoplanet_count,
            'model_type': MODEL_TYPE
        })
        
    except Exception as e:
        import traceback
        print(f"Error in batch_predict: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'loaded': False,
            'message': 'Model not loaded'
        })
    
    return jsonify({
        'loaded': True,
        'model_type': MODEL_TYPE,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
        'architecture': 'CNN feature extraction + LSTM temporal modeling'
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_type': MODEL_TYPE if model is not None else None,
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HYBRID CNN+LSTM EXOPLANET DETECTION WEB UI")
    print("="*70)
    
    if not os.path.exists('templates'):
        print("\nERROR: templates/ directory not found!")
        sys.exit(1)
    
    if not os.path.exists('templates/index_enhanced.html'):
        print("\nERROR: templates/index_enhanced.html not found!")
        print("Create it or rename your existing template file")
        sys.exit(1)
    
    model_loaded = load_model()
    preprocessor_loaded = load_preprocessor()
    
    if not model_loaded:
        print("\nWARNING: Model not loaded!")
        print("Train first: python3 hybrid_training_complete.py\n")
    
    print("\n" + "="*70)
    print("SERVER STARTING")
    print("="*70)
    print(f"Model: {'Loaded' if model_loaded else 'Not loaded'}")
    print(f"Type: {MODEL_TYPE}")
    print(f"Preprocessor: {'Loaded' if preprocessor_loaded else 'Default'}")
    print("="*70)
    print("\nOpen browser: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  / - Main interface")
    print("  POST /predict - Single prediction")
    print("  POST /batch_predict - Batch predictions")
    print("  GET  /model_info - Model information")
    print("  GET  /health - Health check")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nServer stopped")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()