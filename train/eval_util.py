import pandas as pd
from sklearn.metrics import classification_report

from data_util import label_dictionary_to_label_mat


def create_classification_report(test_split, test_preds, CONFIG):
    label_names = label_dictionary_to_label_mat(test_preds.pred).columns.tolist()
    class_report_dict = classification_report(
        label_dictionary_to_label_mat(test_split[CONFIG["label_col"]]),
        label_dictionary_to_label_mat(test_preds.pred),
        target_names=label_names,
        output_dict=True,
    )
    return (
        pd.DataFrame(class_report_dict)
        .T.reset_index()
        .rename(mapper={"index": "label"}, axis="columns", inplace=False)
    )


def create_slim_classification_report(classification_report):
    label_names = [
        e
        for e in classification_report.label
        if e not in ["micro avg", "macro avg", "weighted avg", "samples avg"]
    ]
    return (
        classification_report.query("label in @label_names")
        .pipe(lambda x: x[["label", "f1-score", "support"]])
        .set_index("label")
        .to_dict(orient="index")
    )
