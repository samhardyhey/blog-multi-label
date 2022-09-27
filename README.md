## Multi-label
Notebooks and scripts for the:

- Collection of [AusFinance](https://www.reddit.com/r/AusFinance/) and adjacent reddit data
- Multi-label annotation
- Model spot-checking
- Model deployment

See the accompanying blog post [here](https://www.samhardyhey.com/poor-mans-asr-pt-1). Additionally, see WandB hosted model evaluation here, and HF Gradio toy-deployment [here](https://huggingface.co/spaces/samhardyhey/blog-multi-label).

## Install
- Conda env creation, python dependencies via `create_env.sh`

## Usage
- **Reddit scraping.** Given a collection of subreddits, retrieve all submissions and accompanying comments within a `day_delta` period of time. Ceiling defined via `total_submission_limit`. Run via `python scrape/scrape_reddit.py`. Adjust config to suit.

- **Multi-label annotation.** Given a CSV of formatted submissions/comments, bootstrap a small multi-label dataset using rule-based dictionaries, thresholding and quota requirements. Annotation script allows for successive annotation rounds to be applied to a single dataset, allowing for flexible label scheme changes (add/remove/redefine). Run via `python annotate/annotate_multi_label.py`. Adjust config to suit.

- **Multi-label training.** Given a CSV of annotated submissions/comments, spot-check a selection of models including the original dictionary classifier, an sklearn linear SVC and a flair TARS few-shot classifier. Log model performance/files to WandB. Assumes an active WandB account (`wandb login`). Run via `python train/train_multi_label.py`. Adjust config to suit.

- **Model deployment.** Given a specific annotation round, copy relevant model files into the sub-moduled gradio space repo. Copy via `python train/copy_deploy_model_files.py`, "Deploy" via repo updates.

## HF Space Submodule
Misc. commands mainly for my benefit:

```
# add
git submodule add https://huggingface.co/spaces/samhardyhey/blog-multi-label/ deploy

# update
git submodule update --init --recursive

# remove
git submodule deinit
```