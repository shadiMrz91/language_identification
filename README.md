Text language detection using multilingual transformer models. Compares performance of XLM-RoBERTa, LabSE, and Distil-mBERT, with Distil-mBERT achieving 99% accuracy while being optimized for efficiency.
First run requires internet connection to download:
Subsequent runs work offline - models and data are cached locally.

### Quick Start
Prerequisites
Python 3.13
Internet connection (first run only)

### Installation & Run
python -m venv lang_id_env
lang_id_env/bin/activate

pip install -r requirements.txt
VS Code Launch Configuration
.vscode/launch.json file:

```
json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Full Pipeline (train + test)",
      "type": "python",
      "request": "launch",
      "program": "main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  ]
}
```

To run the project:
Ensure your project structure has main.py in the root directory
Open the project in VS Code
Press F5 to execute the complete pipeline



### Performance Results
Model	      Accuracy	Speed	Size	Best For
XLM-RoBERTa	  87%	    162s	>1GB	Baseline
LabSE	      86%	    84s	    ~500MB	Speed
Distil-mBERT  99%	    90s  	~500MB	Accuracy & Efficiency


### Project Structure
```
project/
├── .vscode/
│   └── launch.json         # VS Code launch configuration
├── main.py                 # Main pipeline
├── data_handler.py         # Data management
├── embedding_extractor.py  # Embedding generation
├── classifier.py           # Classification models
├── test_pipeline.py        # Evaluation
├── requirements.txt        # Dependencies
├── data/                   # Downloaded datasets
├── models/                 # LR-labse and LR-distil-mbert models
└── results/                # Generated outputs
```

### Output Files
Automatically saved in /results/:
Model performance metrics (Accuracy, F1 scores)
Misclassified samples (.xlsx)
Embedding visualizations (.png)
Prediction files (.csv)
Execution timing data

### Dataset
Source: Tatoeba (Hugging Face)
Languages: 10 languages (ES, FR, EN, IT, PT, NL, SV, PL, RU, JA)
Samples: 10,000 to 50,000 text samples

### Technical Details
Framework: PyTorch, Transformers, Scikit-learn
Embedding Models: LabSE, Distil-mBERT, XML-Roberta
Classifier: Logistic Regression with hyperparameter tuning
Evaluation: Accuracy, F1-score

