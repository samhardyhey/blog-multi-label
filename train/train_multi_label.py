from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from clear_bow.classifier import DictionaryClassifier
from sklearn.metrics import classification_report
from skmultilearn.model_selection import iterative_train_test_split

import wandb


def label_dictionary_to_label_mat(label_dictionary_list, thresh=0.75):
    return (
        pd.DataFrame.from_records(list(label_dictionary_list))
        .pipe(lambda x: x >= thresh)
        .astype(int)
    )


def label_mat_to_label_dictionary(label_mat):
    return list(label_mat.to_dict(orient="index").values())


def create_multi_label_train_test_splits(
    df: pd.core.frame.DataFrame,
    label_col: str,
    test_size=0.25,
):
    df[label_col] = df[label_col].apply(
        lambda x: eval(x) if type(x) == str else x
    )  # string > dict

    # threshold, iteratively split
    y_df = label_dictionary_to_label_mat(df[label_col])
    y_cols = list(y_df.columns)
    x_df = df.drop(label_col, axis=1)
    x_cols = list(x_df.columns)

    x_train, y_train, x_test, y_test = iterative_train_test_split(
        x_df.values, y_df.astype(int).values, test_size=test_size
    )

    # convert back to label object form
    y_train = label_mat_to_label_dictionary(
        pd.DataFrame(y_train, columns=y_cols))
    y_test = label_mat_to_label_dictionary(
        pd.DataFrame(y_test, columns=y_cols))

    # re-stack x/y
    train = pd.DataFrame(np.column_stack((x_train, y_train))).set_axis(
        labels=x_cols + [label_col], axis="columns", inplace=False
    )

    test = pd.DataFrame(np.column_stack((x_test, y_test))).set_axis(
        labels=x_cols + [label_col], axis="columns", inplace=False
    )
    return train, test


def log_dataframe(run, df, name, description):
    # any type of df within a run
    df_artifact = wandb.Artifact(
        name, type="dataset", description=description
    )
    df_artifact.add(wandb.Table(dataframe=df), name=name)
    run.log_artifact(df_artifact)


if __name__ == "__main__":
    CONFIG = yaml.safe_load(
        Path(
            "/Users/samhardyhey/Desktop/blog/blog-multi-label/training_config.yaml"
        ).read_bytes()
    )

    # 1.1 create splits
    df = pd.read_csv(CONFIG["dataset"])
    train, test = create_multi_label_train_test_splits(
        df, label_col=CONFIG["label_col"], test_size=CONFIG["test_size"]
    )
    test, dev = create_multi_label_train_test_splits(
        test, label_col=CONFIG["label_col"], test_size=CONFIG["test_size"]
    )

    # 1.2 log splits
    with wandb.init(
        project=CONFIG["wandb_project"],
        name="reddit_aus_finance",
        group=CONFIG["wandb_group"],
        entity="cool_stonebreaker",
    ) as run:
        log_dataframe(run, train, "train_split", "Train split")
        log_dataframe(run, dev, "dev_split", "Dev split")
        log_dataframe(run, test, "test_split", "Test split")

    # mock multiple model runs for a single group
    model_config = CONFIG["models"][0]

    dc = DictionaryClassifier(
        classifier_type=model_config["classifier_type"],
        label_dictionary=model_config["label_dictionary"],
    )

    # get train/test performance > log as model artefacts?
    dev_pred = dev.assign(
        pred=lambda x: x[CONFIG["text_col"]].apply(dc.predict_single))
    test_pred = test.assign(
        pred=lambda x: x[CONFIG["text_col"]].apply(dc.predict_single))

    label_names = label_dictionary_to_label_mat(
        test_pred.pred).columns.tolist()
    class_report = (pd.DataFrame(classification_report(
        label_dictionary_to_label_mat(test[CONFIG["label_col"]]),
        label_dictionary_to_label_mat(test_pred.pred),
        target_names=label_names,
        output_dict=True,
    ))
        .T
        .reset_index()
        .rename(mapper={"index": "label"}, axis="columns", inplace=False)
    )

    slim_class_report = (
        class_report
        .query('label in @label_names')
        .pipe(lambda x: x[["label", "f1-score", "support"]])
        .set_index("label")
        .to_dict(orient="index")
    )

    # seperate models as seperate runs
    with wandb.init(
        project=CONFIG["wandb_project"],
        name=model_config['model'],
        group=CONFIG["wandb_group"],
        entity="cool_stonebreaker",
    ) as run:
        wandb.config.model = model_config['model']
        wandb.config.group = CONFIG["wandb_group"]

        # log dev/pred preds
        log_dataframe(run, dev_pred, "dev_preds", "Dev predictions")
        log_dataframe(run, test_pred, "test_preds", "Test predictions")

        run.log(slim_class_report)
        run.summary["test_f1"] = class_report.query(
            'label == "weighted avg"')['f1-score'].iloc[0]
        run.summary["test_support"] = class_report.query(
            'label == "weighted avg"')['support'].iloc[0]

    # seperate models as seperate runs
    with wandb.init(
        project=CONFIG["wandb_project"],
        name="flair_tars",
        group=CONFIG["wandb_group"],
        entity="cool_stonebreaker",
    ) as run:
        wandb.config.model = "flair_tars"
        wandb.config.group = CONFIG["wandb_group"]

        # log dev/pred preds
        log_dataframe(run, dev_pred, "dev_preds", "Dev predictions")
        log_dataframe(run, test_pred, "test_preds", "Test predictions")

        run.log(slim_class_report)
        run.summary["test_f1"] = class_report.query(
            'label == "weighted avg"')['f1-score'].iloc[0]
        run.summary["test_support"] = class_report.query(
            'label == "weighted avg"')['support'].iloc[0]
