import tempfile
from pathlib import Path

import pandas as pd
from clear_bow.classifier import DictionaryClassifier
from flair.data import Corpus, Sentence
from flair.models import TARSClassifier
from flair.tokenization import SegtokTokenizer
from flair.trainers import ModelTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import BinaryRelevance

import wandb
from data_util import log_dataframe
from eval_util import (create_classification_report,
                       create_slim_classification_report,
                       label_dictionary_to_label_mat)


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


def fit_and_log_linear_svc(train_split, dev_split, test_split, CONFIG, model_config):
    with wandb.init(
        project=CONFIG["wandb_project"],
        name=model_config["type"],
        group=CONFIG["wandb_group"],
        entity=CONFIG["wandb_entity"],
    ) as run:
        wandb.config.type = model_config["type"]
        wandb.config.group = CONFIG["wandb_group"]

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

        # predict/evaluate
        test_preds = test_split.assign(
            pred=[
                dict(
                    zip(
                        test_split.columns.values, pipeline.predict([e])[
                            0].toarray()[0]
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


def create_flair_classification_sentence(text, label_object, label_type="class"):
    sentence = Sentence(text, use_tokenizer=SegtokTokenizer())
    for label in [k for k, v in label_object.items() if v > 0]:
        sentence.add_label(label_type, label, 1.0)
    return sentence


def predict_flair_tars(text, flair_tars_model):
    sentence = Sentence(text)
    labels = flair_tars_model.get_current_label_dictionary().get_items()
    flair_tars_model.predict(sentence)
    pred_dict = {
        label: 0.0 for label in labels
    }
    for e in sentence.labels:
        label = e.to_dict()["value"]
        confidence = round(float(e.to_dict()["confidence"]), 2)
        pred_dict[label] = confidence
    return pred_dict


def fit_and_log_flair_tars(train_split, dev_split, test_split, CONFIG, model_config):
    with wandb.init(
        project=CONFIG["wandb_project"],
        name=model_config["type"],
        group=CONFIG["wandb_group"],
        entity=CONFIG["wandb_entity"],
    ) as run:
        wandb.config.type = model_config["type"]
        wandb.config.group = CONFIG["wandb_group"]
        label_type = model_config.get("label_type", 'multi_label_class')

        train_dev = pd.concat([train_split, dev_split], sort=True)
        train_sents = (
            pd.concat([train_split, dev_split], sort=True)
            .apply(
                lambda x: create_flair_classification_sentence(
                    x[CONFIG["text_col"]], x[CONFIG["label_col"]], label_type
                ),
                axis=1,
            )
            .tolist()
        )
        test_sents = test_split.apply(
            lambda x: create_flair_classification_sentence(
                x[CONFIG["text_col"]], x[CONFIG["label_col"]], label_type
            ),
            axis=1,
        ).tolist()

        # make a corpus with train and test split
        corpus = Corpus(train=train_sents, test=test_sents)

        # train a tiny model, with tiny parameters
        tars = TARSClassifier.load("tars-base")

        # 2. make the model aware of the desired set of labels from the new corpus
        tars.add_and_switch_to_new_task(
            task_name=label_type,
            label_dictionary=corpus.make_label_dictionary(label_type),
            label_type=label_type,
            multi_label=True,
        )

        # 3. initialize the text classifier trainer with your corpus
        trainer = ModelTrainer(tars, corpus)

        # 4. train model
        with tempfile.TemporaryDirectory() as artefact_dir:
            trainer.train(
                base_path=artefact_dir,  # path to store the model artifacts
                learning_rate=model_config.get(
                    "learning_rate", 0.02
                ),  # use very small learning rate
                mini_batch_size=model_config.get(
                    "mini_batch_size", 1
                ),  # small mini-batch size since corpus is tiny
                max_epochs=model_config.get("max_epochs", 10),
                save_final_model=model_config.get("max_epochs", False),
            )
            trainer.model.save(
                Path(artefact_dir) / "final-model.pt", checkpoint=False)

        # predict/evaluate
        test_preds = test_split.assign(pred=test_split[CONFIG["text_col"]].apply(
            lambda y: predict_flair_tars(y, tars)))

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
