from clear_bow.classifier import DictionaryClassifier

import wandb
from data_util import log_dataframe
from eval_util import create_classification_report, create_slim_classification_report


def fit_and_log_dictionary_classifier(test_split, CONFIG, model_config):
    with wandb.init(
        project=CONFIG["wandb_project"],
        name=model_config["type"],
        group=CONFIG["wandb_group"],
        entity=CONFIG["wandb_entity"],
    ) as run:
        wandb.config.type = model_config["type"]
        wandb.config.group = CONFIG["wandb_group"]

        # instantiate
        dc = DictionaryClassifier(
            classifier_type=model_config["classifier_type"],
            label_dictionary=model_config["label_dictionary"],
        )

        # predict/evaluate
        test_preds = test_split.assign(
            pred=lambda x: x[CONFIG["text_col"]].apply(dc.predict_single)
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
