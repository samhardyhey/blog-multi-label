import numpy as np
import pandas as pd
import wandb
from skmultilearn.model_selection import iterative_train_test_split


def label_dictionary_to_label_mat(label_dictionary_list, thresh=0.75):
    return (
        pd.DataFrame.from_records(list(label_dictionary_list))
        .pipe(lambda x: x[sorted(x.columns)])
        .pipe(lambda x: x >= thresh)
        .astype(int)
    )


def label_mat_to_label_dictionary(label_mat):
    return list(
        label_mat.pipe(lambda x: x[sorted(x.columns)]).to_dict(orient="index").values()
    )

def filter_label_object(label_object, target_fields):
    return {k:v for k,v in label_object.items() if k in target_fields}


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
    y_train = label_mat_to_label_dictionary(pd.DataFrame(y_train, columns=y_cols))
    y_test = label_mat_to_label_dictionary(pd.DataFrame(y_test, columns=y_cols))

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
    df_artifact = wandb.Artifact(name, type="dataset", description=description)
    df_artifact.add(wandb.Table(dataframe=df), name=name)
    run.log_artifact(df_artifact)
