# 🚀 Exoplanet Detection: Complete Setup Guide

## 📖 Table of Contents
1. [System Requirements](#system-requirements)
2. [Project Structure](#project-structure)
3. [Installation from Scratch](#installation-from-scratch)
4. [Virtual Environment Setup](#virtual-environment-setup)
5. [Training the Model](#training-the-model)
6. [Using the Model](#using-the-model)
7. [Web Interface](#web-interface)
8. [File Structure Explained](#file-structure-explained)
9. [Troubleshooting](#troubleshooting)
10. [Command Reference](#command-reference)

---

## 💻 System Requirements

### Your System (Verified Compatible ✓)
- **CPU**: AMD Ryzen 5 5700U (8 cores, 1.8-4.3 GHz)
- **RAM**: 16GB
- **OS**: Ubuntu (Linux)
- **Python**: 3.8+ (you have 3.13)
- **Storage**: ~10GB free space

### Software Dependencies
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (venv)
- Git (optional, for version control)

### Required Python Packages
```
tensorflow==2.20.0      # Deep learning framework
numpy>=1.26.0           # Numerical computing
pandas>=2.2.2           # Data manipulation
scikit-learn>=1.4.0     # Machine learning tools
matplotlib>=3.9.0       # Plotting
seaborn>=0.13.0         # Statistical visualization
flask==3.0.0            # Web framework
scipy>=1.14.0           # Scientific computing
```

---

## 📁 Project Structure

Your project directory (`/home/zer0/Main/exoplanet-detection/`) is organized as follows:

```
exoplanet-detection/
│
├── 📂 data/                          # Dataset files (16 CSV files)
│   ├── exoTrain.csv                  # Main training data (251MB)
│   ├── exoTest.csv                   # Main test data (28MB)
│   ├── cumulative.csv                # Kepler cumulative catalog
│   ├── oec.csv                       # Open Exoplanet Catalogue
│   └── ... (12 more dataset files)
│
├── 📂 models/                        # Trained models (created after training)
│   ├── exoplanet_cnn_lstm.h5        # Trained neural network (will be ~10-15MB)
│   └── preprocessor_hybrid.pkl      # Data preprocessor state
│
├── 📂 results/                       # Training visualizations (created after training)
│   ├── training_history_hybrid.png  # Learning curves
│   ├── confusion_matrix_hybrid.png  # Performance matrix
│   ├── roc_curve_hybrid.png         # ROC analysis
│   └── sample_predictions_hybrid.png# Example predictions
│
├── 📂 templates/                     # Web interface HTML files
│   └── index_enhanced.html          # Main web UI
│
├── 📂 venv/                          # Python virtual environment
│   ├── bin/                         # Executables (python, pip, etc.)
│   ├── lib/                         # Installed packages
│   └── pyvenv.cfg                   # Environment configuration
│
├── 🐍 hybrid_cnn_lstm_model.py      # Model architecture definition
├── 🐍 hybrid_training.py            # Training script
├── 🐍 hybrid_preprocessing.py       # Data preprocessing functions
├── 🐍 hybrid_webapp.py              # Flask web application
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 MODEL_EXPLANATION.md          # How the model works (this file's companion)
├── 📄 SETUP_GUIDE.md                # This file
│
└── 📄 training_full.log             # Training progress log (created during training)
```

### File Purposes

| File | Purpose | Size |
|------|---------|------|
| `hybrid_cnn_lstm_model.py` | Defines the neural network architecture | ~16 KB |
| `hybrid_training.py` | Main training script | ~13 KB |
| `hybrid_preprocessing.py` | Data cleaning and normalization | ~11 KB |
| `hybrid_webapp.py` | Web interface server | ~11 KB |
| `requirements.txt` | Lists Python dependencies | 1 KB |
| `index_enhanced.html` | Web UI template | ~35 KB |

---

## 🔧 Installation from Scratch

### Step 1: Open Terminal

```bash
# Navigate to your home directory
cd ~

# Or navigate to where you want the project
cd ~/Main
```

### Step 2: Create Project Directory (if starting fresh)

```bash
# Create directory
mkdir -p exoplanet-detection

# Navigate into it
cd exoplanet-detection
```

### Step 3: Check Python Version

```bash
python3 --version
```

**Expected Output**: `Python 3.8.x` or higher (you have 3.13.x ✓)

If Python is not installed:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Step 4: Download/Copy Project Files

If you're starting from scratch, ensure you have these files:
- All Python files (`hybrid_*.py`)
- `requirements.txt`
- `templates/index_enhanced.html`

Your current setup already has all files ✓

---

## 🐍 Virtual Environment Setup

### Why Use a Virtual Environment?

A virtual environment isolates your project's Python packages from the system Python, preventing conflicts and making the project portable.

### Step 1: Create Virtual Environment

```bash
# Make sure you're in the project directory
cd /home/zer0/Main/exoplanet-detection

# Create virtual environment named 'venv'
python3 -m venv venv
```

**What this does**:
- Creates a `venv/` directory
- Installs a clean Python environment
- Isolates packages from system Python

### Step 2: Activate Virtual Environment

```bash
# Activate the environment
source venv/bin/activate
```

**What you'll see**:
```bash
(venv) zer0@computer:~/Main/exoplanet-detection$
        ↑ This indicates virtual environment is active
```

**Important**: Always activate the virtual environment before running scripts!

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**This installs**:
- TensorFlow 2.20.0 (~500 MB)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Flask
- SciPy

**Installation time**: 5-10 minutes depending on internet speed

### Step 5: Verify Installation

```bash
# Test if all packages are installed
python3 -c "import tensorflow, numpy, pandas, sklearn, scipy, matplotlib, seaborn, flask; print('✓ All packages installed successfully!')"
```

**Expected Output**: `✓ All packages installed successfully!`

### Step 6: Verify TensorFlow

```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Available devices:', len(tf.config.list_physical_devices()))"
```

**Expected Output**:
```
TensorFlow version: 2.20.0
Available devices: 1
```

---

## 📊 Data Setup

### Your Data (Already Present ✓)

You already have 16 CSV files in the `data/` directory:
- `exoTrain.csv` (251 MB) - Main training set
- `exoTest.csv` (28 MB) - Main test set
- Additional Kepler/TESS catalogs

### Verify Data

```bash
# Check data files
ls -lh data/*.csv | wc -l
```

**Expected Output**: `16` (you have all files ✓)

### If You Need to Add More Data

Place new CSV files in the `data/` directory:
```bash
cp /path/to/your/new_data.csv data/
```

The training script automatically discovers and uses all CSV files in `data/`.

---

## 🎯 Training the Model

### Step 1: Ensure Virtual Environment is Active

```bash
source venv/bin/activate
```

### Step 2: Verify Model Configuration

```bash
# Check model parameters
python3 -c "from hybrid_cnn_lstm_model import HybridCNNLSTM; m = HybridCNNLSTM(); m.build_model(); print(f'Parameters: {m.model.count_params():,}')"
```

**Expected Output**: `Parameters: 556,801` ✓

### Step 3: Start Training

```bash
python3 hybrid_training.py
```

**What happens**:
1. **[1/7] Discovering datasets** - Finds all CSV files (1 second)
2. **[2/7] Loading datasets** - Loads and validates data (30-60 seconds)
3. **[3/7] Augmenting** - Creates balanced training set (optional, 10-30 seconds)
4. **[4/7] Preprocessing** - Normalizes 32,384 samples (2-5 minutes)
5. **[5/7] Building model** - Creates neural network (5 seconds)
6. **[6/7] Training** - Main training loop (2-3 hours)
7. **[7/7] Evaluating** - Tests on unseen data (1-2 minutes)

### Training Progress

You'll see output like this:

```
======================================================================
HYBRID CNN+LSTM TRAINING
======================================================================

[1/7] Discovering datasets...
Found 16 CSV files

[2/7] Loading datasets...
Loaded 6 valid datasets
Target length: 3197
Final shape: (32384, 3197)
Exoplanets: 10305, Non-exoplanets: 22079

[3/7] Augmenting...
(Optional - may be skipped if class distribution is good)

[4/7] Preprocessing...
Preprocessing light curves...
  Processed 1000/32384 light curves
  Processed 2000/32384 light curves
  ...
  Processed 32000/32384 light curves

Split sizes:
  Train: (22668, 3197, 1)
  Val:   (4858, 3197, 1)
  Test:  (4858, 3197, 1)

[5/7] Building model...
✓ Hybrid CNN+LSTM architecture created!
Total parameters: 556,801

[6/7] Training...
MAXIMUM PERFORMANCE MODE - Full capacity training

Epoch 1/100
709/709 ━━━━━━━━━━━━━━━━━━━━ 150s - loss: 0.8620 - accuracy: 0.5360
Epoch 1: val_auc improved to 0.6573, saving model

Epoch 5/100
Loss: 0.6163 - Acc: 0.5635 - AUC: 0.7128 - Val_AUC: 0.7760

Epoch 10/100
Loss: 0.4325 - Acc: 0.7172 - AUC: 0.8678 - Val_AUC: 0.8892

... (continues for up to 100 epochs or until early stopping)

Epoch 47/100
Restoring model weights from the end of the best epoch: 27.
Epoch 47: early stopping

[7/7] Evaluating...
======================================================================
TEST SET RESULTS
======================================================================
Accuracy:         0.9734 (97.34%)
Precision:        0.9287 (92.87%)
Recall:           0.8923 (89.23%)
AUC-ROC:          0.9856
======================================================================
```

### Training Time

| Component | Time |
|-----------|------|
| Data loading | 1-2 minutes |
| Preprocessing | 3-5 minutes |
| Training (100 epochs) | 2.5-3.5 hours |
| Evaluation | 1-2 minutes |
| **Total** | **~3-4 hours** |

### Monitor Training

In another terminal (while training runs):

```bash
# Watch progress
tail -f /home/zer0/Main/exoplanet-detection/training_full.log

# Check process
ps aux | grep python3

# Check memory usage
free -h

# Check CPU usage
htop  # Press 'q' to quit
```

### What Gets Created After Training

```
models/
├── exoplanet_cnn_lstm.h5          # Trained model (~10-15 MB)
└── preprocessor_hybrid.pkl         # Preprocessor (~100 KB)

results/
├── training_history_hybrid.png     # Training curves
├── confusion_matrix_hybrid.png     # Classification matrix
├── roc_curve_hybrid.png           # ROC analysis
└── sample_predictions_hybrid.png   # Example predictions
```

---

## 🔮 Using the Model

### Option 1: Web Interface (Easiest)

See [Web Interface](#web-interface) section below.

### Option 2: Python Script

Create a file `predict_single.py`:

```python
from hybrid_cnn_lstm_model import HybridCNNLSTM
from hybrid_preprocessing import ExoplanetDataPreprocessor
import numpy as np
import pandas as pd

# Load trained model
print("Loading model...")
model = HybridCNNLSTM()
model.load_model('models/exoplanet_cnn_lstm.h5')

# Load preprocessor
preprocessor = ExoplanetDataPreprocessor()
preprocessor.load_preprocessor('models/preprocessor_hybrid.pkl')

# Load your data (must have 3197 points)
df = pd.read_csv('data/exoTest.csv')
light_curve = df.iloc[0, 1:].values  # Skip label column

# Preprocess
processed = preprocessor.process_new_data(light_curve)

# Predict
probability = model.model.predict(processed, verbose=0)[0][0]
prediction = "Exoplanet Detected!" if probability > 0.5 else "No Exoplanet"

print(f"\nPrediction: {prediction}")
print(f"Confidence: {probability:.2%}")
```

Run it:
```bash
python3 predict_single.py
```

### Option 3: Batch Processing

Create `predict_batch.py`:

```python
from hybrid_cnn_lstm_model import HybridCNNLSTM
import pandas as pd
import numpy as np

# Load model
model = HybridCNNLSTM()
model.load_model('models/exoplanet_cnn_lstm.h5')

# Load batch data
df = pd.read_csv('data/exoTest.csv')
X = df.iloc[:, 1:].values  # All except label column
X = X.reshape(X.shape[0], X.shape[1], 1)

# Predict for all samples
predictions, probabilities = model.predict(X)

# Save results
results = pd.DataFrame({
    'sample_id': range(len(predictions)),
    'prediction': ['Exoplanet' if p==1 else 'No Exoplanet' for p in predictions],
    'probability': probabilities.flatten()
})

results.to_csv('predictions_output.csv', index=False)
print(f"\nProcessed {len(results)} samples")
print(f"Exoplanets detected: {sum(predictions == 1)}")
print(f"Results saved to: predictions_output.csv")
```

---

## 🌐 Web Interface

### Step 1: Start Web Server

```bash
# Ensure virtual environment is active
source venv/bin/activate

# Start Flask server
python3 hybrid_webapp.py
```

**Expected Output**:
```
======================================================================
HYBRID CNN+LSTM EXOPLANET DETECTION WEB UI
======================================================================

✓ Hybrid CNN+LSTM model loaded from models/exoplanet_cnn_lstm.h5
  Input shape: (None, 3197, 1)
  Output shape: (None, 1)
  Parameters: 556,801

======================================================================
SERVER STARTING
======================================================================
Model: Loaded
Type: Hybrid CNN+LSTM
======================================================================

Open browser: http://localhost:5000

Endpoints:
  GET  / - Main interface
  POST /predict - Single prediction
  POST /batch_predict - Batch predictions
  GET  /model_info - Model information
  GET  /health - Health check

Press Ctrl+C to stop
======================================================================
```

### Step 2: Open Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

### Step 3: Use Web Interface

**Single Prediction**:
1. Click "Choose File"
2. Select a CSV file with light curve data
3. Click "Analyze"
4. View results: prediction + probability + visualization

**Batch Prediction**:
1. Upload CSV with multiple rows (each row = one light curve)
2. Click "Batch Analyze"
3. Download results as CSV

### API Endpoints

**Health Check**:
```bash
curl http://localhost:5000/health
```

**Model Info**:
```bash
curl http://localhost:5000/model_info
```

**Single Prediction** (via API):
```bash
curl -X POST http://localhost:5000/predict \
     -F "file=@data/exoTest.csv"
```

### Stop Web Server

Press `Ctrl+C` in the terminal where the server is running.

Or, if running in background:
```bash
sudo lsof -ti:5000 | xargs kill -9
```

---

## 📂 File Structure Explained

### Core Python Files

#### `hybrid_cnn_lstm_model.py` (556,801 parameters)
**Purpose**: Defines the neural network architecture

**Key Components**:
- `HybridCNNLSTM` class
- `build_model()` - Creates neural network
- `train()` - Training loop
- `evaluate()` - Performance metrics
- `predict()` - Make predictions

**When to edit**: If you want to change model architecture (layers, units, etc.)

#### `hybrid_training.py`
**Purpose**: Main training script

**What it does**:
1. Discovers datasets in `data/`
2. Loads and validates CSV files
3. Merges different datasets
4. Augments minority class
5. Preprocesses all data
6. Builds model
7. Trains for 100 epochs
8. Evaluates on test set
9. Generates visualizations

**When to run**: Every time you want to train a new model

#### `hybrid_preprocessing.py`
**Purpose**: Data cleaning and normalization

**Key Functions**:
- `preprocess_single_curve()` - Cleans one light curve
- `normalize_light_curve()` - Robust normalization
- `detrend_light_curve()` - Removes long-term trends
- `remove_outliers()` - Sigma clipping
- `pad_or_truncate()` - Standardizes length

**When to use**: Automatically called by training script

#### `hybrid_webapp.py`
**Purpose**: Flask web application

**Routes**:
- `/` - Main web interface
- `/predict` - Single prediction
- `/batch_predict` - Batch predictions
- `/model_info` - Model metadata
- `/health` - Server health check

**When to run**: After training, when you want to use the web UI

### Data Files

#### `requirements.txt`
Lists all Python package dependencies. Used by:
```bash
pip install -r requirements.txt
```

#### CSV Files in `data/`
- **exoTrain.csv**: Main training set (5,087 samples)
- **exoTest.csv**: Main test set (570 samples)
- **cumulative.csv**: Kepler cumulative KOI catalog
- **oec.csv**: Open Exoplanet Catalogue
- Others: Additional catalogs from Kepler/TESS

**Format**: Each row is a light curve, columns are flux measurements

### Generated Files

#### `models/exoplanet_cnn_lstm.h5`
- Trained neural network weights
- Size: ~10-15 MB
- Created after training completes
- Can be loaded for predictions without retraining

#### `models/preprocessor_hybrid.pkl`
- Saved preprocessor state
- Ensures consistent preprocessing for new data
- Created during training

#### `results/*.png`
Training visualizations:
- **training_history**: Learning curves (loss, accuracy, AUC)
- **confusion_matrix**: TP, TN, FP, FN breakdown
- **roc_curve**: True positive vs false positive rate
- **sample_predictions**: Example predictions with light curves

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "No CSV files found!"

**Solution**:
```bash
# Check data directory
ls -la data/

# Ensure CSV files are present
# If not, add your data files to data/ directory
```

### Issue: Out of Memory During Training

**Solution 1 - Reduce batch size**:
Edit `hybrid_training.py` line 439:
```python
model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=16)
# Changed from 32 to 16
```

**Solution 2 - Close other applications**:
```bash
# Free up memory
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### Issue: Training Very Slow

**Check CPU usage**:
```bash
htop
# Should see python3 using 400-800% CPU (4-8 cores)
```

**If low CPU usage**:
- Close browser and other heavy applications
- Ensure power mode is set to "Performance"

### Issue: Port 5000 Already in Use

**Solution**:
```bash
# Kill process using port 5000
sudo lsof -ti:5000 | xargs kill -9

# Or use different port
# Edit hybrid_webapp.py line 368:
# app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: "Model not found" when starting webapp

**Solution**:
```bash
# Train model first
python3 hybrid_training.py

# Wait for training to complete (~3 hours)
# Then start webapp
python3 hybrid_webapp.py
```

### Issue: Training Interrupted

**Solution**:
Training progress is saved automatically! The best model so far is saved to `models/exoplanet_cnn_lstm.h5`.

To resume or start fresh:
```bash
# Start training again
python3 hybrid_training.py

# It will overwrite with better model if found
```

---

## ⌨️ Command Reference

### Virtual Environment

```bash
# Create
python3 -m venv venv

# Activate
source venv/bin/activate

# Deactivate
deactivate

# Delete and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
# Standard training
python3 hybrid_training.py

# Training in background (lets you close terminal)
nohup python3 hybrid_training.py > training.log 2>&1 &

# Monitor background training
tail -f training.log

# Check if training is running
ps aux | grep hybrid_training
```

### Web Interface

```bash
# Start server
python3 hybrid_webapp.py

# Start in background
nohup python3 hybrid_webapp.py > webapp.log 2>&1 &

# Stop server
# Press Ctrl+C (if running in foreground)
# Or:
sudo lsof -ti:5000 | xargs kill -9
```

### Model Operations

```bash
# Check model parameters
python3 -c "from hybrid_cnn_lstm_model import HybridCNNLSTM; m=HybridCNNLSTM(); m.build_model(); m.summary()"

# Test model loading
python3 -c "from hybrid_cnn_lstm_model import HybridCNNLSTM; m=HybridCNNLSTM(); m.load_model(); print('✓ Model loaded')"

# Check model file
ls -lh models/exoplanet_cnn_lstm.h5
```

### System Monitoring

```bash
# Memory usage
free -h

# CPU usage
htop  # or: top

# Disk space
df -h

# Process list
ps aux | grep python3

# Network (for webapp)
netstat -tlnp | grep 5000
```

### Data Operations

```bash
# Count CSV files
ls -1 data/*.csv | wc -l

# Check CSV file size
ls -lh data/*.csv

# Preview CSV
head -5 data/exoTrain.csv

# Count rows in CSV
wc -l data/exoTrain.csv
```

---

## ✅ Quick Setup Checklist

Use this checklist to ensure everything is set up correctly:

- [ ] Python 3.8+ installed
- [ ] Project directory created
- [ ] Virtual environment created (`venv/`)
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] TensorFlow working (test import)
- [ ] Data files present in `data/` (16 CSV files)
- [ ] Model file present (`hybrid_cnn_lstm_model.py`)
- [ ] Training script present (`hybrid_training.py`)
- [ ] Preprocessing script present (`hybrid_preprocessing.py`)
- [ ] Web app present (`hybrid_webapp.py`)
- [ ] Template present (`templates/index_enhanced.html`)

**When all checked**, you're ready to train!

---

## 🎯 Workflow Summary

### First Time Setup (30 minutes)
```bash
1. cd /home/zer0/Main/exoplanet-detection
2. python3 -m venv venv
3. source venv/bin/activate
4. pip install -r requirements.txt
5. # Ensure data files are in data/
```

### Training (3-4 hours)
```bash
1. source venv/bin/activate
2. python3 hybrid_training.py
3. # Wait for completion
4. # Check results/ directory
```

### Using the Model
```bash
1. source venv/bin/activate
2. python3 hybrid_webapp.py
3. # Open: http://localhost:5000
4. # Upload CSV and get predictions
```

---

## 📞 Need Help?

### Check These First
1. **Logs**: Look at terminal output or `training_full.log`
2. **Virtual Environment**: Is it activated? (`(venv)` in prompt)
3. **Dependencies**: Run `pip list` to see installed packages
4. **Data**: Are CSV files present in `data/`?
5. **Model**: Does `models/exoplanet_cnn_lstm.h5` exist?

### Common Commands
```bash
# Re-verify installation
pip install -r requirements.txt

# Test imports
python3 -c "import tensorflow; print('OK')"

# Check model
python3 -c "from hybrid_cnn_lstm_model import HybridCNNLSTM; print('OK')"

# List data files
ls data/*.csv
```

---

**Setup Complete! You're ready to detect exoplanets!** 🌟🚀

System: Ryzen 5 5700U, 16GB RAM  
Model: 556,801 parameters  
Configuration: Maximum Performance Mode
