import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from psaw import PushshiftAPI
from tqdm import tqdm

if __name__ == "__main__":
    TOTAL_SUBMISSION_LIMIT = 250
    DAY_DELTA = 60
    SUBREDDITS = [
        "fiaustralia",
        "ASX_Bets",
        "ausstocks",
        "AusProperty",
        "AusFinance",
        "ausstocks",
        "AusEcon",
        "AusPropertyChat",
        "ASX",
        "AustralianAccounting",
    ]
    OUTPUT_DIR = Path("./output")
    shutil.rmtree((str(OUTPUT_DIR))) if OUTPUT_DIR.exists() else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pushshift_client = PushshiftAPI()
    last_month_start_epoch = int(
        (datetime.now() - timedelta(days=DAY_DELTA)).timestamp()
    )
    reddit_query = ""
    per_subreddit_limit = math.ceil(TOTAL_SUBMISSION_LIMIT / len(SUBREDDITS))

    # 1. retrieve submissions
    all_subreddit_submissions = []
    for subreddit in tqdm(
        SUBREDDITS,
        desc=f"Collecting {per_subreddit_limit} submissions for each subreddit..",
    ):
        submission_raw = list(
            pushshift_client.search_submissions(
                q=reddit_query,
                after=last_month_start_epoch,
                subreddit=subreddit,
                filter=[
                    "url",
                    "author",
                    "id",
                    "parent_id",
                    "link_id",
                    "title",
                    "subreddit",
                ],
                limit=per_subreddit_limit,
            )
        )
        submissions_formatted = pd.DataFrame([e.d_ for e in submission_raw])
        all_subreddit_submissions.append(submissions_formatted)

    all_subreddit_submissions = pd.concat(all_subreddit_submissions)

    # 1. retrieve related comments
    submissions_and_comments = []
    for idx, record in tqdm(
        all_subreddit_submissions.iterrows(),
        total=all_subreddit_submissions.shape[0],
        desc="Collecting submission comments..",
    ):
        comments_raw = list(
            pushshift_client.search_comments(
                after=last_month_start_epoch,
                subreddit=record.subreddit,
                link_id=record.id,
                filter=[
                    "url",
                    "author",
                    "id",
                    "parent_id",
                    "title",
                    "body",
                    "subreddit",
                ],
            )
        )
        comments_formatted = pd.DataFrame([e.d_ for e in comments_raw])

        submissions_and_comments.append(
            pd.concat([record.to_frame().transpose(), comments_formatted], sort=True)
        )

    # 3. format/save
    all_submissions_and_comments = (
        pd.concat(submissions_and_comments, sort=True)
        .assign(
            document_publish_date=lambda x: x.created.apply(
                lambda y: datetime.fromtimestamp(y)
            )
        )
        .drop(labels=["created", "created_utc"], axis="columns", inplace=False)
    )
    all_submissions_and_comments.to_csv(
        OUTPUT_DIR / "aus_finance_reddit.csv", index=False
    )
