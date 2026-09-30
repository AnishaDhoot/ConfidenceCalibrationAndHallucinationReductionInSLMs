# VIT Q&A Project

A VIT question-answering project with dataset preparation, evaluation, fine-tuning utilities, and a Flask chatbot demo site.

## Run the chatbot site

From the project root:

```powershell
pip install -r chatbot_site/requirements.txt
python chatbot_site/app.py
```

Open `http://127.0.0.1:5000` in a browser.

The default chatbot backend uses the prepared dataset. To use a LoRA adapter instead:

```powershell
$env:CHATBOT_BACKEND = "lora"
$env:ADAPTER_PATH = "phi3-vit-lora-expc-idk15"
python chatbot_site/app.py
```

## Project contents

- `chatbot_site/`: Flask application and web interface
- `prepare_expc_data.py`: Prepare experiment datasets
- `train_lora.py`: LoRA fine-tuning workflow
- `verify_dataset.py`: Dataset validation
- `eval_compare_fixed.py` and calibration scripts: Evaluation utilities
- `*.jsonl`, `*.csv`: Datasets and evaluation files
