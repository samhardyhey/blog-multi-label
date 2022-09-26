import shutil
from pathlib import Path

import wandb
import yaml
from wasabi import msg

from eval_util import list_all_project_artifacts

ANNOTATION_GROUP = "annotation_1"

if __name__ == "__main__":
    api = wandb.Api()
    CONFIG = yaml.safe_load((Path(__file__).parents[0] / "train_config.yaml").read_bytes())

    # get run ID's via artifacts
    project_artifacts = list_all_project_artifacts(api, CONFIG)
    output_base_dir = Path(__file__).parents[1] / "deploy/model_files"
    run_records = project_artifacts.query("group == @ANNOTATION_GROUP").drop_duplicates("type")

    for idx, record in run_records.iterrows():
        # save each model's binaries/config
        if record.type == "flair_tars":
            msg.info("Saving flair_tars model files..")
            output_dir = output_base_dir / "flair_tars"
            if output_dir.exists():
                shutil.rmtree(str(output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            wandb.restore(
                name="final-model.pt",
                run_path="/".join(record.run.path),
                replace=True,
                root=str(output_dir),
            )

        elif record.type == "sklearn_linear_svc":
            msg.info("Saving sklearn_linear_svc model files..")
            output_dir = output_base_dir / "sklearn_linear_svc"
            if output_dir.exists():
                shutil.rmtree(str(output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            wandb.restore(
                name="model.joblib",
                run_path="/".join(record.run.path),
                replace=True,
                root=str(output_dir),
            )

        elif record.type == "dictionary_classifier":
            msg.info("Saving dictionary_classifier model files..")
            output_dir = output_base_dir / "dictionary_classifier"
            if output_dir.exists():
                shutil.rmtree(str(output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            wandb.restore(
                name="label_dictionary.json",
                run_path="/".join(record.run.path),
                replace=True,
                root=str(output_dir),
            )
