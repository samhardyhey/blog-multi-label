import copy
import itertools
import os
from pathlib import Path

import pandas as pd
import yaml
from clear_bow.classifier import DictionaryClassifier
from wasabi import msg


def binary_confirm_n_label_objects(
    df,
    label_col,
    label_object_key,
    accept_value="y",
    reject_value="n",
    n_examples=10,
):
    # given a list of records, positively verify (binary confirmation) across a selected field until examples run out/quota reached. Return all annotations.
    updated_records = []
    msg.info(
        f"Input: '{accept_value}' to accept value, input {reject_value} to reject value. "
        + f"Label objects saved as '{label_col}' field within all records. "
        + f"Loop will break when '{n_examples}' positively affirmed or examples run out, whichever first. {df.shape[0]} candidates supplied.\n"
    )
    for idx, record in df.iterrows():
        # early break if n_examples exists
        if (
            len(
                [
                    record
                    for record in updated_records
                    if record[label_col][label_object_key] == 1
                ]
            )
            >= n_examples
        ):
            return pd.DataFrame(updated_records)

        # otherwise, verify n_examples
        d = record.to_dict()
        dc_d = copy.deepcopy(d)  # always use a deep copy
        _ = [print("\033[1m", k, ": ", "\033[0m", v) for k, v in d.items()]
        val = input(f"Instance of {label_object_key}? ")
        verification_remapping = {"y": 1, "n": -1, "": 0.0, " ": 0.0}
        dc_d[label_col][label_object_key] = verification_remapping.get(val)
        updated_records.append(dc_d)
        os.system("clear")
    return pd.DataFrame(updated_records)


def consolidate_hard_soft_labels(label_objects):
    # defer to hard labels where they exist, otherwise average soft labels
    label_objects = [object for object in label_objects if type(object) == dict]

    flattened_object = {}
    all_labels = pd.DataFrame(label_objects)
    for label in all_labels:
        # hard reject
        if -1 in set(all_labels[label]):
            flattened_object[label] = -1
        # hard accept
        elif 1 in set(all_labels[label]):
            flattened_object[label] = 1
        # mean otherwise
        else:
            flattened_object[label] = all_labels[label].mean()

    return flattened_object


def consolidate_doc_labels(df, label_col):
    label_objects = df[label_col].tolist()

    if len([object for object in label_objects if type(object) == dict]) == 0:
        # NAN values - usually from concatenated, unseen records
        df[label_col] = [None] * df.shape[0]
    else:
        df[label_col] = [consolidate_hard_soft_labels(label_objects)] * df.shape[0]
    return df.head(1)


def annotate_n_examples_per_class(
    model,
    df,
    text_col,
    n_examples=10,
    specific_labels=[],
    prediction_thresh=0.75,
    rank_candidates=True,
    max_candidates=50,
    label_col="label",
):
    # 1. label book-keeping - restrict against a pre-existing set of labels
    model_labels = [
        label
        for label in model.predict_single("hello world").keys()
        if label != "no_label"
    ]
    if specific_labels:
        # sanity check the specified labels against available model labels
        label_intersection = sorted(
            list(set(specific_labels).intersection(set(model_labels)))
        )
        if not label_intersection:
            raise ValueError("No label intersection found, aborting sampling")
        msg.info(f"Specified: {sorted(specific_labels)}")
        msg.info(f"Model features: {sorted(model_labels)}")
        msg.info(f"Sampling using the intersecting labels only: {label_intersection}")
        pred_labels = label_intersection
    else:
        pred_labels = model_labels

    if label_col in df:
        # 3.1 consolidate old verifications with new predictions, defer to "hard" old predictions where they exist
        consolidated_labels = []
        for hard, soft in zip(
            df[label_col].tolist(), df[text_col].apply(model.predict_single).tolist()
        ):
            if type(hard) != dict:
                # RE: iteratively appending datasets
                consolidated_labels.append(soft)
            else:
                consolidated_labels.append(consolidate_hard_soft_labels([hard, soft]))
        df[label_col] = consolidated_labels
    else:
        # 3.2 otherwise create new prediction
        df[label_col] = df[text_col].apply(model.predict_single)

    all_verifications = []
    for label in pred_labels:
        # 4. only use unseen examples, if verification field exists
        seen_examples = None
        if label_col in df:
            seen_examples = df[
                df[label_col].apply(lambda y: True if type(y[label]) == int else False)
            ]
            unseen_examples = df[
                ~df[label_col].apply(lambda y: True if type(y[label]) == int else False)
            ]
            if seen_examples.shape[0] > 1:
                msg.info(
                    f"{seen_examples.shape[0]} pre-existing, positive examples found for label: '{label}'"
                )

        # reduce if necessary
        adjusted_n_examples = (
            n_examples - seen_examples.shape[0]
            if seen_examples.shape[0] > 0
            else n_examples
        )

        # 5. only annotate examples with relatively high confidence
        annotate_input = (
            unseen_examples
            # only examine examples above prediction threshold
            .pipe(
                lambda x: x[
                    x[label_col].apply(
                        lambda y: True if y[label] >= prediction_thresh else False
                    )
                ]
            )
        )

        if rank_candidates:
            # 6.1 rank by prediction confidence
            annotate_input = annotate_input.reset_index(drop=True).pipe(
                lambda x: x.iloc[
                    x[label_col]
                    .apply(lambda x: x[label])
                    .sort_values(ascending=False)
                    .index
                ]
            )
        else:
            # 6.2 shuffle candidates otherwise
            annotate_input = annotate_input.sample(frac=1.0, random_state=42)

        # 7. take the first n=max_candidate records
        annotate_input = annotate_input.head(max_candidates)

        # 8. actually annotate the filtered data..
        if annotate_input.shape[0] == 0:
            msg.warn(
                f"\n\n****\t No candidate examples found for: {label}, skipping\t****\n"
            )
            annotations = pd.DataFrame()
        else:
            msg.info(f"\n\n****\t Annotating: {label} \t****\n")
            annotations = binary_confirm_n_label_objects(
                annotate_input, label_col, label, n_examples=adjusted_n_examples
            )

        # 9. consolidate alongside seen examples; grow pool of annotations
        if seen_examples.shape[0] > 1:
            annotations = pd.concat([seen_examples, annotations])

        all_verifications.append(annotations)

    # 10. consolidate across records
    return (
        pd.concat(all_verifications)
        .groupby(text_col)
        .apply(lambda x: consolidate_doc_labels(x, label_col))
        .reset_index(drop=True)
    )


def backfill_multi_label_objects(
    model, df, text_col, label_col, focus_labels=None, prediction_thresh=0.75
):
    # Given an original set of labels/predictions (labels_a), verify and backfill complimentary model predictions (labels_b)
    if focus_labels is None:
        focus_labels = []
    df = df.copy(deep=True)

    # generally assume entire label space
    label_space_b = [
        label
        for label in model.predict_single("hello world").keys()
        if label != "no_label"
    ]
    if focus_labels:
        # optionally reduce label space to specific sub-set of existing labelspace
        target_label_space = set(focus_labels).intersection(set(label_space_b))
    else:
        target_label_space = set(label_space_b)

    assert target_label_space  # at least one label
    target_label_space = sorted(list(target_label_space))

    consolidated_labels = []
    for hard, soft in zip(
        df[label_col].tolist(), df[text_col].apply(model.predict_single).tolist()
    ):
        # using fresh predictions, consolidate label space
        if type(hard) != dict:
            # RE: iteratively appending datasets
            consolidated_labels.append(soft)
        else:
            consolidated_labels.append(consolidate_hard_soft_labels([hard, soft]))

    for label, (idx, label_object) in itertools.product(
        target_label_space, enumerate(consolidated_labels)
    ):
        if label_object[label] in [-1, 1]:
            # 1. pre-existing hard label, no change
            continue

        elif label_object[label] >= prediction_thresh:
            # 3. otherwise, some difference in labels, as proposed by model
            os.system("clear")
            msg.info(f"**** Verifying all additional instances of: {label} ****")
            msg.text(f"\n\033[1mText: \033[0m \n{df.iloc[idx][text_col]}\n")
            msg.text(f"\033[1mInstance of {label}?\033[0m")
            confirmation = input()

            if confirmation == "n":
                label_object[label] = -1

            elif confirmation == "y":
                # assign in place.. yikes
                label_object[label] = 1

    df[label_col] = consolidated_labels
    return df


if __name__ == "__main__":
    CONFIG = yaml.safe_load(
        (Path(__file__).parents[0] / "annotation_config.yaml").read_bytes()
    )
    df = (
        pd.read_csv(CONFIG["input_dataset"])
        .pipe(
            lambda x: x[
                x[CONFIG["text_col"]].apply(
                    lambda y: len(y.split(" ")) > CONFIG["min_tokens"]
                )
            ]
        )
        .pipe(
            lambda x: x[
                x[CONFIG["text_col"]].apply(
                    lambda y: len(y.split(" ")) < CONFIG["max_tokens"]
                )
            ]
        )
        .drop_duplicates(CONFIG["text_col"])
    )

    if CONFIG["label_col"] in df:
        # probably a pre-existing dataset
        df[CONFIG["label_col"]] = df[CONFIG["label_col"]].apply(eval)

    # 1. use a cheap, dictionary classifier to bootstrap initial, per-class annotations
    dc = DictionaryClassifier(
        classifier_type=CONFIG["classifier_type"],
        label_dictionary=CONFIG["label_dictionary"],
    )

    annotations = annotate_n_examples_per_class(
        model=dc,
        df=df,
        text_col=CONFIG["text_col"],
        n_examples=CONFIG["n_examples"],
        prediction_thresh=CONFIG["prediction_threshold"],
        rank_candidates=CONFIG["rank_candidates"],
        max_candidates=CONFIG["max_candidates"],
        label_col=CONFIG["label_col"],
    )

    # 2. backfill across multi-label space
    annotations = backfill_multi_label_objects(
        model=dc,
        df=annotations,
        text_col=CONFIG["text_col"],
        label_col=CONFIG["label_col"],
        prediction_thresh=CONFIG["prediction_threshold"],
    )

    annotations.to_csv(
        Path(CONFIG["output_dir"]) / f"{CONFIG['group']}.csv", index=False
    )
