# Legal Clause Semantic Similarity

Deep learning project for semantic similarity detection in legal clauses using Siamese BiLSTM and Attention-based Encoder models.

## Project Structure

```
.
├── archive/              # Dataset CSV files
├── preprocessing.py      # Data preprocessing module
├── model.ipynb          # Model training and evaluation notebook
├── requirements.txt     # Python dependencies
├── setup_venv.bat      # Windows setup script
├── setup_venv.sh       # Linux/Mac setup script
└── README.md           # This file
```

## Setup Instructions

### ⚠️ Important: Python Version Requirements

**TensorFlow requires Python 3.8-3.12.** Python 3.14 is not yet supported.

If you have Python 3.14, please:
1. Install Python 3.11 from [python.org](https://www.python.org/downloads/)
2. Use Python 3.11 to create the virtual environment (see below)

### Windows

1. **Create and activate virtual environment:**
   ```bash
   # Option 1: Use Python 3.11 (if installed)
   py -3.11 -m venv venv
   
   # Option 2: Use setup script (requires Python 3.8-3.12)
   setup_venv.bat
   
   # Option 3: Manual setup
   python -m venv venv  # Only if Python 3.8-3.12
   venv\Scripts\activate.bat
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Activate virtual environment (for future sessions):**
   ```bash
   venv\Scripts\activate.bat
   # Or for PowerShell:
   venv\Scripts\Activate.ps1
   ```
   
   **If you get execution policy error in PowerShell:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Linux/Mac

1. **Create and activate virtual environment:**
   ```bash
   # Option 1: Use Python 3.11 (if installed)
   python3.11 -m venv venv
   
   # Option 2: Use the setup script (requires Python 3.8-3.12)
   chmod +x setup_venv.sh
   ./setup_venv.sh
   
   # Option 3: Manual setup
   python3 -m venv venv  # Only if Python 3.8-3.12
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Activate virtual environment (for future sessions):**
   ```bash
   source venv/bin/activate
   ```

## Usage

1. **Activate the virtual environment:**
   - Windows: `venv\Scripts\activate.bat`
   - Linux/Mac: `source venv/bin/activate`

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open and run `model.ipynb`:**
   - The notebook will automatically import functions from `preprocessing.py`
   - Run all cells sequentially to train and evaluate both models

## Dependencies

### Required for preprocessing.py
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **tensorflow**: Deep learning framework (for text tokenization)

### Required for model.ipynb (full project)
- All above dependencies
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **scikit-learn**: Only for evaluation metrics (accuracy, precision, recall, etc.)
- **jupyter**: Notebook environment

### Important Notes
- ✅ **preprocessing.py does NOT require sklearn/scikit-learn**
- The `train_test_split` function has been replaced with a custom implementation using only numpy
- sklearn is only used in the notebook for evaluation metrics
- See `DEPENDENCIES.md` for more details

## Output Files

After running the notebook, you will have:

- `pairs.csv`: Generated clause pairs with labels
- `tokenizer.json`: Saved tokenizer for text preprocessing
- `bilstm_model.h5`: Trained Siamese BiLSTM model
- `attention_model.h5`: Trained Attention-based Encoder model
- `training_curves.png`: Training/validation curves
- `confusion_matrices.png`: Confusion matrices for both models

## Models

### 1. Siamese BiLSTM
- Shared Embedding layer (128 dimensions)
- Shared Bidirectional LSTM (128 units)
- Combination using [u, v, |u-v|, u*v]
- Dense layers with dropout

### 2. Attention-based Encoder
- Shared Embedding layer (128 dimensions)
- Shared Bidirectional LSTM with return sequences
- Self-attention layer
- Same combination and dense layers

## Notes

- The dataset files should be in the `archive/` directory
- Models are trained with early stopping (patience=4)
- Training uses batch size of 64 and up to 25 epochs
- Random seeds are set for reproducibility (seed=42)

## Deactivate Virtual Environment

When you're done working:
```bash
deactivate
```

