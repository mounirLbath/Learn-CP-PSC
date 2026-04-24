"""
ml_service.py
=============
Bridge between the Django app and the CFRecommender.

Responsibilities
----------------
1. Load user_problem_table.csv from the project root.
2. Pull the connected user's stored submissions from the DB and append
   them as new rows, so the model always sees the freshest data.
3. Train all three models (KNN, SGD, LASSO) and the ensemble.
4. Return ranked candidate lists per model for the connected user.

The recommender is re-trained on every page load. 
"""

import os
import sys
import json
import pandas as pd

from django.conf import settings
from .models import CodeforcesUser, UserSubmission
from cf_recommender import _parse_tags


# Path to the CSV file sitting next to manage.py
CSV_PATH = os.path.join(settings.BASE_DIR, 'codeforces_submissions.csv') 

# Number of recommendations to surface per model
TOP_K = 10


def _load_csv() -> pd.DataFrame:
    """
    Load the base training dataset from user_problem_table.csv.

    Expected columns (case-insensitive, extra columns ignored):
        USER, USER_ELO, PROBLEM, PROBLEM_ELO, PROBLEM_TAGS, VERDICT

    PROBLEM_TAGS may be a JSON string ("[\"dp\",\"greedy\"]") or a
    Python list — both are normalised to a list.
    """
    df = pd.read_csv(CSV_PATH)

    # Normalise column names to uppercase
    df.columns = [c.strip().upper() for c in df.columns]

    required = {'USER', 'USER_ELO', 'PROBLEM', 'PROBLEM_ELO', 'PROBLEM_TAGS', 'VERDICT'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"codeforces_submissions.csv is missing columns: {missing}")

    df['PROBLEM_TAGS'] = df['PROBLEM_TAGS'].apply(_parse_tags)
    df['USER_ELO']     = pd.to_numeric(df['USER_ELO'],     errors='coerce').fillna(1200).astype(int)
    df['PROBLEM_ELO']  = pd.to_numeric(df['PROBLEM_ELO'],  errors='coerce').fillna(1200).astype(int)

    return df


def _user_submissions_to_rows(handle: str) -> list[dict]:
    """
    Fetch the connected user's stored submissions from the DB and convert
    them to rows compatible with the CSV schema.

    USER_ELO is left as 0 here — it will be filled from the CSV if the
    user already appears there, or left as a neutral default otherwise.
    """
    try:
        user = CodeforcesUser.objects.get(handle__iexact=handle)
    except CodeforcesUser.DoesNotExist:
        return []

    rows = []
    for sub in UserSubmission.objects.filter(user=user).values(
        'problem_name', 'problem_index', 'contest_id',
        'problem_rating', 'problem_tags', 'verdict',
    ):
        # Build a problem identifier consistent with what might already be
        # in the CSV: "contestId + index"  e.g. "1234A"
        problem_id = (
            f"{sub['contest_id']}{sub['problem_index']}"
            if sub['contest_id']
            else sub['problem_name'] or 'UNKNOWN'
        )
        rows.append({
            'USER':         handle,
            'USER_ELO':     0,           # placeholder; overridden below
            'PROBLEM':      problem_id,
            'PROBLEM_ELO':  sub['problem_rating'] or 1200,
            'PROBLEM_TAGS': _parse_tags(sub['problem_tags']),
            'VERDICT':      sub['verdict'] or 'UNKNOWN',
        })
    return rows


def _merge_user_data(base_df: pd.DataFrame, handle: str) -> pd.DataFrame:
    """
    Append the connected user's live submissions to the base DataFrame.

    If the user already appears in the CSV (from historical data), their
    USER_ELO is taken from there. Otherwise 1200 is used as a neutral default.
    """
    new_rows = _user_submissions_to_rows(handle)
    if not new_rows:
        return base_df

    # Resolve USER_ELO from existing data if available
    existing_elo = base_df.loc[
        base_df['USER'].str.lower() == handle.lower(), 'USER_ELO'
    ]
    user_elo = int(existing_elo.iloc[0]) if not existing_elo.empty else 1200

    for row in new_rows:
        row['USER_ELO'] = user_elo

    new_df = pd.DataFrame(new_rows)
    merged = pd.concat([base_df, new_df], ignore_index=True)

    # Drop exact duplicates (same USER + PROBLEM + VERDICT)
    merged = merged.drop_duplicates(subset=['USER', 'PROBLEM', 'VERDICT'])

    return merged


def run_ml_recommender(handle: str, top_k: int = TOP_K) -> dict:
    """
    Full pipeline: load CSV → merge user submissions → train → recommend.

    Returns a dict with keys:
        handle      — the user's CF handle
        top_k       — number of candidates per model
        knn         — list of ranked problem dicts from KNN
        sgd         — list of ranked problem dicts from SGD
        lasso       — list of ranked problem dicts from LASSO
        ensemble    — list of ranked problem dicts from the ensemble
        tag_prefs   — LASSO tag preference dict (positive = strength, negative = weakness)
        error       — error message string if something went wrong, else None
    """
    # Add project root to sys.path so we can import cf_recommender.py
    project_root = str(settings.BASE_DIR)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from cf_recommender import CFRecommender
    except ImportError as e:
        return _error_result(handle, f"Could not import cf_recommender.py: {e}")

    try:
        base_df = _load_csv()
    except FileNotFoundError:
        return _error_result(handle, f"codeforces_submissions.csv not found at {CSV_PATH}")
    except ValueError as e:
        return _error_result(handle, str(e))

    df = _merge_user_data(base_df, handle)

    # The user must appear in the dataset to get recommendations
    if handle.lower() not in df['USER'].str.lower().values:
        return _error_result(handle, f"No data found for user '{handle}' in the dataset.")

    try:
        rec = CFRecommender(df, target_prob_lo=0.50,missing_strategy="ignore", target_prob_hi=0.65,lasso_print_tags=True) # Strategies : "zero", "ignore", "elo_fill"
        rec.fit(nb_neighbors=20, nb_components=20, nb_iterations=15)
    except Exception as e:
        return _error_result(handle, f"Model training failed: {e}")

    def _rank(model_name):
        try:
            df_rank = rec.rank_candidates(handle, model=model_name, top_k=top_k)
            if df_rank.empty:
                return []
            return df_rank.to_dict('records')
        except Exception as e:
            return [{'error': str(e)}]

    tag_prefs = {}
    try:
        tag_prefs = rec.explain(handle)
    except Exception:
        pass

    return {
        'handle':    handle,
        'top_k':     top_k,
        'knn':       _rank('knn'),
        #'sgd':       _rank('sgd'),
        'lasso':     _rank('lasso'),
        'ensemble':  _rank('ensemble'),
        'tag_prefs': tag_prefs,
        'error':     None,
    }


def _error_result(handle: str, message: str) -> dict:
    return {
        'handle':    handle,
        'top_k':     TOP_K,
        'knn':       [],
        'sgd':       [],
        'lasso':     [],
        'ensemble':  [],
        'tag_prefs': {},
        'error':     message,
    }