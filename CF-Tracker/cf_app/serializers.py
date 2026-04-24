from rest_framework import serializers
from .models import CodeforcesUser, UserSubmission, UserStatusFetch


class UserSubmissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSubmission
        fields = [
            'id',
            'submission_id',
            'contest_id',
            'problem_index',
            'problem_name',
            'problem_rating',
            'problem_tags',
            'creation_time_seconds',
            'programming_language',
            'verdict',
            'time_consumed_millis',
            'memory_consumed_bytes',
            'raw_data',
        ]


class UserStatusFetchSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserStatusFetch
        fields = ['id', 'fetched_at', 'status', 'submissions_count', 'error_message']


class CodeforcesUserSerializer(serializers.ModelSerializer):
    submissions = UserSubmissionSerializer(many=True, read_only=True)
    fetches = UserStatusFetchSerializer(many=True, read_only=True)
    submission_count = serializers.SerializerMethodField()

    class Meta:
        model = CodeforcesUser
        fields = ['id', 'handle', 'created_at', 'updated_at', 'submission_count', 'submissions', 'fetches']

    def get_submission_count(self, obj):
        return obj.submissions.count()


class FetchUserStatusSerializer(serializers.Serializer):
    """Input serializer: just the Codeforces handle."""
    handle = serializers.CharField(
        max_length=100,
        help_text="Codeforces user handle (e.g. 'tourist')"
    )
    from_index = serializers.IntegerField(
        default=1,
        min_value=1,
        help_text="1-based index of the first submission to fetch (default: 1)"
    )
    count = serializers.IntegerField(
        default=100,
        min_value=1,
        max_value=10000,
        help_text="Number of submissions to fetch (default: 100, max: 10000)"
    )
