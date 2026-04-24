import json
import random

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views import View

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import CodeforcesUser, UserSubmission
from .serializers import FetchUserStatusSerializer, CodeforcesUserSerializer, UserSubmissionSerializer
from .services import (
    fetch_user_rating,
    store_user_status,
    recommend_problem,
    fetch_contest_problems,
    compute_elo,
    CodeforcesAPIError,
    _parse_tags,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

SESSION_KEY = 'cf_handle'

# Tags shown on the dashboard. Each gets its own Elo rating and recommender.
TRACKED_TAGS = [
    'implementation', 'math', 'greedy', 'dp', 'data structures',
    'brute force', 'constructive algorithms', 'graphs', 'sortings',
    'binary search', 'dfs and similar', 'trees', 'strings',
    'number theory', 'combinatorics',
]


# ─────────────────────────────────────────────────────────────────────────────
#  Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_session_handle(request) -> str | None:
    """Return the Codeforces handle stored in the current session, or None."""
    return request.session.get(SESSION_KEY)


def login_required_handle(view_func):
    """
    Decorator for View methods.
    Redirects to the login page if no handle is stored in the session.
    """
    def wrapper(self, request, *args, **kwargs):
        if not get_session_handle(request):
            return redirect('cf_app:login')
        return view_func(self, request, *args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
#  Template context helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tag_stats(user) -> list:
    """
    Compute an Elo rating and solved-problem count for each tracked tag.
    Fetches all rated submissions in a single query, parses tags in Python
    (to avoid JSONField lookup inconsistencies across SQLite/PostgreSQL),
    then runs compute_elo() per tag.
    Returns a list of dicts: tag, tag_slug, solved_count, elo.
    """
    all_subs = list(
        UserSubmission.objects.filter(user=user)
        .exclude(problem_rating__isnull=True)
        .order_by('creation_time_seconds')
        .values('problem_rating', 'problem_tags', 'verdict', 'creation_time_seconds')
    )
 
    # Parse tags once up front to avoid repeated JSON decoding
    parsed_subs = [
        {**sub, 'problem_tags': _parse_tags(sub['problem_tags'])}
        for sub in all_subs
    ]
 
    # Use the user's actual CF rating as the starting point for the Elo simulation.
    user_rating = fetch_user_rating(user.handle) or 1200
 
    stats = []
    for tag in TRACKED_TAGS:
        tag_subs = [s for s in parsed_subs if tag in s['problem_tags']]
        elo = compute_elo(tag_subs, initial=user_rating)
        solved_count = len({
            (s['problem_rating'],)  # approximate; good enough for display
            for s in tag_subs if s['verdict'] == 'OK'
        })
        stats.append({
            'tag':          tag,
            'tag_slug':     tag.replace(' ', '-'),
            'solved_count': solved_count,
            'elo':          elo,
        })
    return stats

def _base_ctx(request) -> dict:
    """
    Build the template context shared by all logged-in views.
    Includes the session handle, total submission count, and per-tag stats.
    """
    handle = get_session_handle(request)
    user = CodeforcesUser.objects.filter(handle__iexact=handle).first()
    return {
        'handle':           handle,
        'submission_count': user.submissions.count() if user else 0,
        'tag_stats':        _tag_stats(user) if user else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Auth views
# ─────────────────────────────────────────────────────────────────────────────

class LoginView(View):
    """
    GET  /login/  — show the handle entry form.
    POST /login/  — validate the handle against CF, fetch submissions, start session.
    Redirects to the dashboard if a session is already active.
    """

    def get(self, request):
        if get_session_handle(request):
            return redirect('cf_app:index')
        return render(request, 'cf_app/login.html')

    def post(self, request):
        handle = request.POST.get('handle', '').strip()
        if not handle:
            return render(request, 'cf_app/login.html', {'error': 'Please enter your Codeforces handle.'})

        try:
            result = store_user_status(handle, count=1000)
        except CodeforcesAPIError as e:
            return render(request, 'cf_app/login.html', {'error': str(e)})

        request.session[SESSION_KEY] = result['user'].handle
        return redirect('cf_app:index')


class LogoutView(View):
    """POST /logout/  — clear the session and redirect to login."""

    def post(self, request):
        request.session.pop(SESSION_KEY, None)
        return redirect('cf_app:login')


class ChangeHandleView(View):
    """
    POST /change-handle/  — switch to a different CF handle.
    Re-fetches 1000 submissions for the new handle and updates the session.
    """

    def post(self, request):
        handle = request.POST.get('handle', '').strip()
        if not handle:
            return redirect('cf_app:index')

        try:
            result = store_user_status(handle, count=1000)
        except CodeforcesAPIError as e:
            return render(request, 'cf_app/index.html', {
                **_base_ctx(request),
                'message': str(e),
                'message_type': 'error',
            })

        request.session[SESSION_KEY] = result['user'].handle
        return redirect('cf_app:index')


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class IndexView(View):
    """GET /  — main dashboard showing per-tag Elo stats and the global recommender."""

    @login_required_handle
    def get(self, request):
        return render(request, 'cf_app/index.html', _base_ctx(request))


class RefreshSubmissionsView(View):
    """POST /refresh/  — re-fetch the latest 1000 submissions for the current user."""

    @login_required_handle
    def post(self, request):
        handle = get_session_handle(request)
        try:
            result = store_user_status(handle, count=1000)
        except CodeforcesAPIError as e:
            return render(request, 'cf_app/index.html', {
                **_base_ctx(request),
                'message': str(e),
                'message_type': 'error',
            })

        return render(request, 'cf_app/index.html', {
            **_base_ctx(request),
            'message': f"Submissions refreshed — {result['total_fetched']} fetched, {result['created_count']} new.",
            'message_type': 'success',
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Recommender
# ─────────────────────────────────────────────────────────────────────────────

class RecommendProblemView(View):
    """
    GET /recommend/?tags=<tag>  — return a JSON problem recommendation.
    If ?tags is provided, the recommendation is scoped to that tag and anchored
    on the user's tag-specific Elo. Without ?tags, uses the global max solved rating.
    """

    @login_required_handle
    def get(self, request):
        handle = get_session_handle(request)
        tag = request.GET.get('tags', '').strip() or None

        try:
            problem = recommend_problem(handle, tag=tag)
        except CodeforcesAPIError as e:
            return JsonResponse({'error': str(e)}, status=404)

        return JsonResponse({
            'problem_name':  problem.get('name'),
            'problem_index': problem.get('index'),
            'contest_id':    problem.get('contestId'),
            'rating':        problem.get('rating'),
            'tags':          problem.get('tags', []),
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Submissions history
# ─────────────────────────────────────────────────────────────────────────────

class SubmissionsView(View):
    """
    GET /submissions/  — paginated submission history for the current user.
    Supports filtering by verdict (?verdict=ok|fail) and tag (?tag=dp).
    Tag filtering is done in Python because JSONField containment queries
    behave inconsistently across SQLite and PostgreSQL.
    """

    @login_required_handle
    def get(self, request):
        handle = get_session_handle(request)
        user = CodeforcesUser.objects.filter(handle__iexact=handle).first()

        page           = max(1, int(request.GET.get('page', 1)))
        per_page       = 50
        verdict_filter = request.GET.get('verdict', '')
        tag_filter     = request.GET.get('tag', '')

        qs = UserSubmission.objects.filter(user=user).order_by('-creation_time_seconds')

        if verdict_filter == 'ok':
            qs = qs.filter(verdict='OK')
        elif verdict_filter == 'fail':
            qs = qs.exclude(verdict='OK')

        fields = (
            'id', 'submission_id', 'problem_name', 'problem_index',
            'contest_id', 'problem_rating', 'problem_tags',
            'verdict', 'programming_language', 'creation_time_seconds',
            'time_consumed_millis',
        )

        if tag_filter:
            # Pull all matching rows then filter by tag in Python
            all_subs = list(qs.values(*fields))
            submissions_all = [
                s for s in all_subs
                if tag_filter in _parse_tags(s['problem_tags'])
            ]
            total = len(submissions_all)
            start = (page - 1) * per_page
            submissions = submissions_all[start:start + per_page]
        else:
            total = qs.count()
            start = (page - 1) * per_page
            submissions = list(qs[start:start + per_page].values(*fields))

        total_pages = max(1, (total + per_page - 1) // per_page)

        return render(request, 'cf_app/submissions.html', {
            **_base_ctx(request),
            'submissions':    submissions,
            'total':          total,
            'page':           page,
            'total_pages':    total_pages,
            'per_page':       per_page,
            'verdict_filter': verdict_filter,
            'tag_filter':     tag_filter,
            'tracked_tags':   TRACKED_TAGS,
        })


# ─────────────────────────────────────────────────────────────────────────────
#  Contest planner
# ─────────────────────────────────────────────────────────────────────────────

class ContestView(View):
    """
    GET  /contest/  — show the contest ID input form.
    POST /contest/  — fetch all problems for the given contest ID and
                      randomly recommend one to start with.
                      The recommendation logic is a placeholder; replace it
                      with smarter logic (e.g. Elo-based) when ready.
    """

    @login_required_handle
    def get(self, request):
        return render(request, 'cf_app/contest.html', {
            **_base_ctx(request),
            'problems': None, 'recommended': None, 'contest_id': '',
        })

    @login_required_handle
    def post(self, request):
        contest_id_raw = request.POST.get('contest_id', '').strip()

        if not contest_id_raw.isdigit():
            return render(request, 'cf_app/contest.html', {
                **_base_ctx(request),
                'error': 'Please enter a valid numeric contest ID.',
                'problems': None, 'recommended': None, 'contest_id': contest_id_raw,
            })

        contest_id = int(contest_id_raw)

        try:
            problems = fetch_contest_problems(contest_id)
        except CodeforcesAPIError as e:
            return render(request, 'cf_app/contest.html', {
                **_base_ctx(request),
                'error': str(e),
                'problems': None, 'recommended': None, 'contest_id': contest_id_raw,
            })

        recommended = random.choice(problems) if problems else None

        return render(request, 'cf_app/contest.html', {
            **_base_ctx(request),
            'contest_id':  contest_id_raw,
            'problems':    problems,
            'recommended': recommended,
        })




# ─────────────────────────────────────────────────────────────────────────────
#  ML Recommender view
# ─────────────────────────────────────────────────────────────────────────────

class MLRecommendView(View):
    """
    GET /ml-recommend/
    Loads user_problem_table.csv, merges the connected user's live submissions,
    trains KNN / SGD / LASSO / Ensemble, and renders a ranked candidate list
    per model. Runs automatically on page load.
    """

    @login_required_handle
    def get(self, request):
        from .ml_service import run_ml_recommender
        handle  = get_session_handle(request)
        results = run_ml_recommender(handle)

        # Post-process candidate rows for template rendering
        for key in ('knn', 'sgd', 'lasso', 'ensemble'):
            for row in results.get(key, []):
                prob = row.get('solve_prob', 0)
                # Percentage width for the probability bar (capped 0-100)
                row['solve_prob_pct']     = round(min(max(prob * 100, 0), 100), 1)
                row['solve_prob_display'] = f"{prob * 100:.1f}%"
                # Parse contest_id and problem_index from the problem string e.g. "1234A"
                problem_str = row.get('problem', '')
                import re
                m = re.match(r'^(\d+)([A-Z]\d*)$', problem_str)
                row['contest_id']    = m.group(1) if m else None
                row['problem_index'] = m.group(2) if m else None

        # Pass models as an ordered list so the template can iterate
        results['models'] = [
            ('ensemble', results.get('ensemble', [])),
            ('sgd',      results.get('sgd',      [])),
            ('knn',      results.get('knn',       [])),
            ('lasso',    results.get('lasso',     [])),
        ]

        return render(request, 'cf_app/ml_recommend.html', {
            **_base_ctx(request),
            **results,
        })

# ─────────────────────────────────────────────────────────────────────────────
#  REST API  (consumed by external tools / the original test scripts)
# ─────────────────────────────────────────────────────────────────────────────

class FetchUserStatusView(APIView):
    """POST /api/fetch/  — trigger a submission fetch for any handle."""

    def post(self, request):
        serializer = FetchUserStatusSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        handle     = serializer.validated_data['handle']
        from_index = serializer.validated_data['from_index']
        count      = serializer.validated_data['count']

        try:
            result = store_user_status(handle, from_index, count)
        except CodeforcesAPIError as e:
            return Response({'error': str(e)}, status=status.HTTP_502_BAD_GATEWAY)

        return Response({
            'message':             f"Successfully fetched submissions for '{handle}'.",
            'handle':              handle,
            'total_fetched':       result['total_fetched'],
            'new_submissions':     result['created_count'],
            'updated_submissions': result['updated_count'],
        })


class UserListView(APIView):
    """GET /api/users/  — list all tracked users with basic stats."""

    def get(self, request):
        users = CodeforcesUser.objects.all().order_by('handle')
        data = [
            {
                'id':               u.id,
                'handle':           u.handle,
                'submission_count': u.submissions.count(),
                'last_fetch':       u.fetches.filter(status='success')
                                     .values_list('fetched_at', flat=True).first(),
            }
            for u in users
        ]
        return Response(data)


class UserDetailView(APIView):
    """GET /api/users/<handle>/  — full detail for a single user."""

    def get(self, request, handle):
        try:
            user = CodeforcesUser.objects.get(handle__iexact=handle)
        except CodeforcesUser.DoesNotExist:
            return Response({'error': f"User '{handle}' not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(CodeforcesUserSerializer(user).data)


class UserSubmissionsView(APIView):
    """
    GET /api/users/<handle>/submissions/
    Returns stored submissions for a user. Supports ?verdict, ?lang, ?contest_id filters.
    """

    def get(self, request, handle):
        try:
            user = CodeforcesUser.objects.get(handle__iexact=handle)
        except CodeforcesUser.DoesNotExist:
            return Response({'error': f"User '{handle}' not found."}, status=status.HTTP_404_NOT_FOUND)

        qs = UserSubmission.objects.filter(user=user)
        verdict    = request.query_params.get('verdict')
        lang       = request.query_params.get('lang')
        contest_id = request.query_params.get('contest_id')

        if verdict:    qs = qs.filter(verdict=verdict.upper())
        if lang:       qs = qs.filter(programming_language__icontains=lang)
        if contest_id: qs = qs.filter(contest_id=contest_id)

        serializer = UserSubmissionSerializer(qs, many=True)
        return Response({'handle': user.handle, 'count': qs.count(), 'submissions': serializer.data})