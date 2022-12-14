from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import BinaryRelevance

import wandb
from data_util import log_dataframe
from eval_util import (
    create_classification_report,
    create_slim_classification_report,
    label_dictionary_to_label_mat,
)


def fit_and_log_sklearn_linear_svc_classifier(
    train_split, dev_split, test_split, CONFIG, model_config
):
    with wandb.init(
        project=CONFIG["wandb_project"],
        name=model_config["type"],
        group=CONFIG["wandb_group"],
        entity=CONFIG["wandb_entity"],
    ) as run:
        wandb.config.type = model_config["type"]

        # define a basic pipeline
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("vt", VarianceThreshold()),
                ("linear_svc", BinaryRelevance(LinearSVC())),
            ]
        )

        # fit
        train_dev = pd.concat([train_split, dev_split], sort=True)
        pipeline.fit(
            train_dev[CONFIG["text_col"]],
            label_dictionary_to_label_mat(train_dev[CONFIG["label_col"]]),
        )
        pipeline.multi_label_classes_ = label_dictionary_to_label_mat(
            test_split[CONFIG["label_col"]]
        ).columns.values
        dump(pipeline, Path(run.dir) / "model.joblib")

        # predict/evaluate
        test_preds = test_split.assign(
            pred=[
                dict(
                    zip(
                        label_dictionary_to_label_mat(
                            test_split[CONFIG["label_col"]]
                        ).columns.values,
                        pipeline.predict([e])[0].toarray()[0],
                    )
                )
                for e in test_split[CONFIG["text_col"]].tolist()
            ]
        )

        classification_report = create_classification_report(
            test_split, test_preds, CONFIG
        )
        classification_report_slim = create_slim_classification_report(
            classification_report
        )

        # log
        log_dataframe(run, test_preds, "test_preds", "Test predictions")
        log_dataframe(
            run,
            classification_report,
            "test_classification_report",
            "Test classification report",
        )
        run.log(classification_report_slim)
        run.summary["test_f1"] = classification_report.query('label == "weighted avg"')[
            "f1-score"
        ].iloc[0]
        run.summary["test_support"] = classification_report.query(
            'label == "weighted avg"'
        )["support"].iloc[0]
