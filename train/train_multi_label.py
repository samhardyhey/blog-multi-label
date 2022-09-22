from pathlib import Path

import pandas as pd
import yaml

import wandb
from data_util import create_multi_label_train_test_splits
from eval_util import (
    list_all_project_artifacts,
    log_inter_group_model_comparisons,
    log_intra_group_model_comparisons,
)
from model.dictionary import fit_and_log_dictionary_classifier
from model.linear_svc import fit_and_log_linear_svc_classifier

api = wandb.Api()

if __name__ == "__main__":
    CONFIG = yaml.safe_load(
        (Path(__file__).parents[0] / "train_config.yaml").read_bytes()
    )

    # 1. create/log splits
    df = pd.read_csv(CONFIG["dataset"])
    train_split, test_split = create_multi_label_train_test_splits(
        df, label_col=CONFIG["label_col"], test_size=CONFIG["test_size"]
    )
    test_split, dev_split = create_multi_label_train_test_splits(
        test_split, label_col=CONFIG["label_col"], test_size=CONFIG["test_size"]
    )
    # with wandb.init(
    #     project=CONFIG["wandb_project"],
    #     name="reddit_aus_finance",
    #     group=CONFIG["wandb_group"],
    #     entity="cool_stonebreaker",
    # ) as run:
    #     log_dataframe(run, train_split, "train_split", "Train split")
    #     log_dataframe(run, dev_split, "dev_split", "Dev split")
    #     log_dataframe(run, test_split, "test_split", "Test split")

    # 2. train/log a selection of models
    for model_config in CONFIG["models"]:
        if model_config["type"] == "dictionary_classifier":
            fit_and_log_dictionary_classifier(
                test_split=test_split, CONFIG=CONFIG, model_config=model_config
            )

        elif model_config["type"] == "sklearn_linear_svc":
            fit_and_log_linear_svc_classifier(
                train_split=train_split,
                dev_split=dev_split,
                test_split=test_split,
                CONFIG=CONFIG,
                model_config=model_config,
            )

        # if model_config["type"] == "flair_tars":
        #     fit_and_log_flair_tars_classifier(
        #         train_split=train_split,
        #         dev_split=dev_split,
        #         test_split=test_split,
        #         CONFIG=CONFIG,
        #         model_config=model_config,
        #     )

        else:
            print(f"Unsupported model: {model_config['type']} found")

    # 3. log intra-model comparisons for current group
    project_artifacts = list_all_project_artifacts(api, CONFIG)
    log_intra_group_model_comparisons(project_artifacts, CONFIG)

    # 4. update inter-group model comparisons
    _ = [
        run.delete()
        for run in api.runs(path="cool_stonebreaker/tyre_kick")
        if run.name == "inter_group_model_comparisons"
    ]
    log_inter_group_model_comparisons(project_artifacts, CONFIG)