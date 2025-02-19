# Multi-Label Text Classification 🏷️

Tools for building and evaluating multi-label text classifiers using Reddit data. Companion code for ["A Perfectly Cromulent Multi-Label Text Classifier"](https://www.samhardyhey.com/a-perfectly-cromulent-multi-label-text-classifier).

## Features
- 🤖 Reddit data collection
- ✍️ Multi-label annotation workflow
- 📊 Model evaluation & comparison
- 🚀 Gradio deployment

## Setup
```bash
# Install dependencies
./create_env.sh
```

## Usage
### 📥 Data Collection
```bash
# Scrape Reddit submissions and comments
python scrape/scrape_reddit.py
```

### 🏷️ Annotation
```bash
# Bootstrap multi-label dataset
python annotate/annotate_multi_label.py
```

### 🔬 Training
```bash
# Evaluate models and log to WandB
wandb login
python train/train_multi_label.py
```

### 🚀 Deployment
```bash
# Copy model files to Gradio space
python train/copy_deploy_model_files.py
```

## Resources
- 📊 [Model Evaluation (WandB)](https://wandb.ai/)
- 🎮 [Demo (Hugging Face)](https://huggingface.co/spaces/samhardyhey/blog-multi-label)

*Note: Configs can be adjusted for each step to suit your needs.*