import tempfile
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import srsly
import wandb
from sklearn.metrics import classification_report

from data_util import label_dictionary_to_label_mat


def create_classification_report(test_split, test_preds, CONFIG):
    label_names = label_dictionary_to_label_mat(test_preds.pred).columns.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def list_all_project_artifacts(api, CONFIG):
    runs_artifacts = (
        pd.DataFrame(
            [
                {**{"run": run}, **run.__dict__["_attrs"]}
                for run in api.runs(
                    path=f"{CONFIG['wandb_entity']}/{CONFIG['wandb_project']}"
                )
            ]
        )
        .assign(
            artifacts=lambda x: x.run.apply(
                lambda y: [
                    {**{"artifact": e}, **e.__dict__} for e in y.logged_artifacts()
                ]
            )
        )
        .pipe(lambda x: x.explode("artifacts"))
        .reset_index(drop=True)
        # finshed runs and config as a proxy for completeness
        .query('state == "finished"')
        .pipe(lambda x: x[x.config.apply(lambda y: len(y) >= 1)])
    )

    return pd.concat(
        [
            runs_artifacts,
            runs_artifacts.config.apply(pd.Series),
            runs_artifacts.artifacts.apply(pd.Series),
        ],
        axis=1,
    ).pipe(lambda x: x[["run", "type", "group", "_sequence_name", "artifact"]])


def parse_wandb_table_artifact(artifact):
    # some real ugly temp dir/json code
    with tempfile.TemporaryDirectory() as temp_dir:
        download_dir = artifact.download(temp_dir)
        file = list(Path(download_dir).glob("*.json"))[0]
        table_json = srsly.read_json(file)
        return pd.DataFrame(table_json["data"], columns=table_json["columns"])


def log_intra_group_model_comparisons(project_artifacts, CONFIG):
    group_model_classification_reports = []

    # format, concat
    for idx, record in (
        project_artifacts.query('_sequence_name == "test_classification_report"').pipe(
            lambda x: x[x.group == CONFIG["wandb_group"]]
        )
    ).iterrows():
        group_model_classification_reports.append(
            (parse_wandb_table_artifact(record.artifact).assign(type=record.type))
        )
    group_model_classification_reports = pd.concat(group_model_classification_reports)

    # create plot
    fig = px.bar(
        (
            group_model_classification_reports.pipe(
                lambda x: x[~x["label"].str.contains("accuracy|samples|macro|micro")]
            )
        ),
        x="label",
        y="f1-score",
        color="type",
        barmode="group",
    )

    # log plot
    with wandb.init(
        project=CONFIG["wandb_project"],
        name=f"{CONFIG['wandb_group']}_intra_group_model_comparison",
        group=CONFIG["wandb_group"],
        entity=CONFIG["wandb_entity"],
        job_type="aux_plot",
    ) as run:
        run.log({f"{CONFIG['wandb_group']}_intra_model_comparison": fig})


def get_most_performant_classifier_per_group(group):
    type = (
        group.query('label == "weighted avg"')
        .sort_values("f1-score", ascending=False)
        .iloc[0]
        .type
    )
    return group.query("type == @type").assign(type=f"{group.iloc[0].group}_{type}")


def log_inter_group_model_comparisons(project_artifacts, CONFIG):
    group_model_classification_reports = []
    # format, concat
    for idx, record in (
        project_artifacts.query('_sequence_name == "test_classification_report"')
    ).iterrows():
        group_model_classification_reports.append(
            (
                parse_wandb_table_artifact(record.artifact)
                .assign(type=record.type)
                .assign(group=record.group)
            )
        )
    group_model_classification_reports = pd.concat(group_model_classification_reports)

    # choose single most performant model from each group
    group_model_classification_reports = (
        group_model_classification_reports.groupby("group")
        .apply(get_most_performant_classifier_per_group)
        .reset_index(drop=True)
    )

    # create plot
    fig = px.bar(
        (
            group_model_classification_reports.pipe(
                lambda x: x[~x["label"].str.contains("accuracy|samples|macro|micro")]
            )
        ),
        x="label",
        y="f1-score",
        color="type",
        barmode="group",
    )

    # log plot
    with wandb.init(
        project=CONFIG["wandb_project"],
        name="inter_group_model_comparison",
        group="inter_group_model_comparison",
        entity=CONFIG["wandb_entity"],
        job_type="aux_plot",
    ) as run:
        run.log({"inter_group_model_comparison": fig})
