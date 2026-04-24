import json
import time
import random
import requests
from django.conf import settings
from .models import CodeforcesUser, UserSubmission, UserStatusFetch

_user_rating_cache: dict = {}
CF_API_BASE = getattr(settings, 'CODEFORCES_API_BASE', 'https://codeforces.com/api')

# In-memory cache for the full CF problem set. Refreshed every hour.
# Avoids hammering the API on every recommendation request.
_problemset_cache = {
    'problems': None,
    'fetched_at': 0,
    'ttl': 3600,
}


class CodeforcesAPIError(Exception):
    """Raised when the Codeforces API returns a non-OK response or is unreachable."""
    pass


def _cf_get(url: str, params: dict = None, timeout: int = 10) -> dict:
    """
    Thin wrapper around requests.get for Codeforces API calls.
    Handles network errors and non-OK API responses uniformly.
    Returns the parsed JSON result on success.
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise CodeforcesAPIError(f"Codeforces API timed out ({url}).")
    except requests.exceptions.RequestException as e:
        raise CodeforcesAPIError(f"Network error contacting Codeforces: {e}")

    data = response.json()
    if data.get('status') != 'OK':
        raise CodeforcesAPIError(f"Codeforces API error: {data.get('comment', 'Unknown')}")

    return data['result']


def _parse_tags(raw) -> list:
    """
    Safely parse a problem_tags value that may be a Python list (from JSONField)
    or a JSON-encoded string (from older SQLite storage).
    """
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return []


def compute_elo(submissions: list, K: int = 32, initial: int = 1200):
    """
    Run a standard Elo simulation over a list of submission dicts,
    processed in chronological order (oldest first).

    Each submission is a match between the user and the problem:
      - verdict OK  -> win  (score = 1)
      - any other   -> loss (score = 0)

    Only submissions with a known problem_rating are counted.
    Returns the final Elo as an int, or None if no rated submissions were found.
    """
    elo = float(initial)
    matched = False

    for sub in sorted(submissions, key=lambda s: s.get('creation_time_seconds', 0)):
        rating = sub.get('problem_rating')
        if rating is None:
            continue
        matched = True
        expected = 1 / (1 + 10 ** ((rating - elo) / 400))
        actual   = 1.0 if sub.get('verdict') == 'OK' else 0.0
        elo += K * (actual - expected)

    return round(elo) if matched else None


def fetch_problemset() -> list:
    """
    Fetch and cache the full Codeforces problem set (~10k problems).
    Held in memory and refreshed at most once per hour.
    Used by the recommender to pick candidates without storing problems locally.
    """
    now = time.time()
    if (
        _problemset_cache['problems'] is not None
        and now - _problemset_cache['fetched_at'] < _problemset_cache['ttl']
    ):
        return _problemset_cache['problems']

    result = _cf_get(f"{CF_API_BASE}/problemset.problems", timeout=15)
    problems = result['problems']
    _problemset_cache['problems'] = problems
    _problemset_cache['fetched_at'] = now
    return problems


def fetch_contest_problems(contest_id: int) -> list:
    """
    Fetch the problem list for a specific contest.
    Uses contest.standings with count=1 to retrieve the problem list cheaply
    without downloading the full leaderboard.
    Returns a list of problem dicts: index, name, rating (optional), tags.
    """
    result = _cf_get(
        f"{CF_API_BASE}/contest.standings",
        params={'contestId': contest_id, 'from': 1, 'count': 1, 'showUnofficial': False},
    )
    return result['problems']


def fetch_user_rating(handle: str) -> int | None:
    """
    Fetch the current CF rating for a user via the user.info API.
    Returns the rating as an int, or None if the user is unrated.
    Cached in memory per handle for the lifetime of the process.
    """
    if handle in _user_rating_cache:
        return _user_rating_cache[handle]
    try:
        result = _cf_get(f"{CF_API_BASE}/user.info", params={'handles': handle})
        rating = result[0].get('rating')
        _user_rating_cache[handle] = rating
        return rating
    except CodeforcesAPIError:
        return None
 

def fetch_user_status(handle: str, from_index: int = 1, count: int = 1000) -> list:
    """
    Fetch raw submission history for a user via the CF User.status API.
    Returns a list of submission dicts as returned by the CF API.
    """
    return _cf_get(
        f"{CF_API_BASE}/user.status",
        params={'handle': handle, 'from': from_index, 'count': count},
    )


def store_user_status(handle: str, from_index: int = 1, count: int = 1000) -> dict:
    """
    Fetch a user's submission history from Codeforces and persist it to the DB.
    Creates the CodeforcesUser record if it does not exist yet.
    Uses update_or_create keyed on submission_id, so repeated calls are idempotent.

    Returns a dict with:
        user            - CodeforcesUser instance
        fetch_log       - UserStatusFetch audit record
        created_count   - new submissions inserted
        updated_count   - existing submissions updated
        total_fetched   - total submissions returned by the API
    """
    user, _ = CodeforcesUser.objects.get_or_create(handle=handle)

    try:
        submissions = fetch_user_status(handle, from_index, count)
    except CodeforcesAPIError as e:
        UserStatusFetch.objects.create(
            user=user, status='failed', submissions_count=0, error_message=str(e),
        )
        raise

    created_count = 0
    updated_count = 0

    for sub in submissions:
        problem = sub.get('problem', {})
        defaults = {
            'contest_id':            sub.get('contestId'),
            'problem_index':         problem.get('index', ''),
            'problem_name':          problem.get('name', ''),
            'problem_rating':        problem.get('rating'),
            'problem_tags':          problem.get('tags', []),
            'creation_time_seconds': sub.get('creationTimeSeconds', 0),
            'programming_language':  sub.get('programmingLanguage', ''),
            'verdict':               sub.get('verdict'),
            'time_consumed_millis':  sub.get('timeConsumedMillis', 0),
            'memory_consumed_bytes': sub.get('memoryConsumedBytes', 0),
            'raw_data':              sub,
        }
        _, created = UserSubmission.objects.update_or_create(
            user=user, submission_id=sub['id'], defaults=defaults,
        )
        if created:
            created_count += 1
        else:
            updated_count += 1

    fetch_log = UserStatusFetch.objects.create(
        user=user, status='success', submissions_count=len(submissions),
    )

    return {
        'user':          user,
        'fetch_log':     fetch_log,
        'created_count': created_count,
        'updated_count': updated_count,
        'total_fetched': len(submissions),
    }


def recommend_problem(handle: str, tag: str = None) -> dict:
    """
    Pick a random unsolved problem suited to the user's current level.
 
    With a tag:
      - Compute the user's tag-specific Elo via compute_elo().
      - Target problems with that tag rated in [tag_elo, tag_elo + 200].
      - If the window is empty, widen to [tag_elo - 200, tag_elo + 400].
      - If still empty, fall back to any unsolved rated problem with that tag.
 
    Without a tag:
      - Use the highest rating the user has ever solved as the anchor.
      - Target any unsolved problem in [anchor, anchor + 200].
      - Fall back to any unsolved rated problem if the window is empty.
 
    Problems the user has already attempted (any verdict) are always excluded.
    """
    try:
        user = CodeforcesUser.objects.get(handle__iexact=handle)
    except CodeforcesUser.DoesNotExist:
        raise CodeforcesAPIError(f"User '{handle}' not found.")
 
    # Exclude every problem already attempted, regardless of verdict.
    attempted = set(
        UserSubmission.objects.filter(user=user)
        .values_list('contest_id', 'problem_index')
    )
 
    problems = fetch_problemset()
 
    def is_candidate(p):
        """True if the problem is unsolved and has a known rating."""
        return (
            (p.get('contestId'), p.get('index')) not in attempted
            and p.get('rating') is not None
        )
 
    if tag:
        # Fetch all rated submissions and filter to those carrying the target tag.
        all_subs = list(
            UserSubmission.objects.filter(user=user)
            .exclude(problem_rating__isnull=True)
            .order_by('creation_time_seconds')
            .values('problem_rating', 'problem_tags', 'verdict', 'creation_time_seconds')
        )
        tag_subs = [s for s in all_subs if tag in _parse_tags(s['problem_tags'])]
        user_rating = fetch_user_rating(handle)
        anchor = compute_elo(tag_subs, initial=user_rating or 1200)
 
        # Primary window: [tag_elo, tag_elo + 200]
        candidates = [
            p for p in problems
            if is_candidate(p)
            and tag in p.get('tags', [])
            and anchor is not None
            and anchor <= p['rating'] <= anchor + 200
        ]
        # Wider fallback: [tag_elo - 200, tag_elo + 400]
        if not candidates:
            lo = (anchor - 200) if anchor else 800
            hi = (anchor + 400) if anchor else 2000
            candidates = [
                p for p in problems
                if is_candidate(p)
                and tag in p.get('tags', [])
                and lo <= p['rating'] <= hi
            ]
        # Last resort: any unsolved rated problem with the tag
        if not candidates:
            candidates = [p for p in problems if is_candidate(p) and tag in p.get('tags', [])]
 
    else:
        # Global mode: anchor on the highest rating the user has solved.
        anchor = (
            UserSubmission.objects.filter(user=user, verdict='OK')
            .exclude(problem_rating__isnull=True)
            .order_by('-problem_rating')
            .values_list('problem_rating', flat=True)
            .first()
        )
        candidates = [
            p for p in problems
            if is_candidate(p)
            and (anchor is None or anchor <= p['rating'] <= anchor + 200)
        ]
        if not candidates:
            candidates = [p for p in problems if is_candidate(p)]
 
    if not candidates:
        raise CodeforcesAPIError("No unsolved problems found matching the criteria.")
 
    return random.choice(candidates)