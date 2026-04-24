from django.db import models


class CodeforcesUser(models.Model):
    """Stores a Codeforces user handle."""
    handle = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.handle


class UserSubmission(models.Model):
    """Stores a single submission from User.status API response."""

    VERDICT_CHOICES = [
        ('OK', 'Accepted'),
        ('WRONG_ANSWER', 'Wrong Answer'),
        ('TIME_LIMIT_EXCEEDED', 'Time Limit Exceeded'),
        ('MEMORY_LIMIT_EXCEEDED', 'Memory Limit Exceeded'),
        ('RUNTIME_ERROR', 'Runtime Error'),
        ('COMPILATION_ERROR', 'Compilation Error'),
        ('CHALLENGED', 'Challenged'),
        ('FAILED', 'Failed'),
        ('PARTIAL', 'Partial'),
        ('PRESENTATION_ERROR', 'Presentation Error'),
        ('IDLENESS_LIMIT_EXCEEDED', 'Idleness Limit Exceeded'),
        ('SECURITY_VIOLATED', 'Security Violated'),
        ('CRASHED', 'Crashed'),
        ('INPUT_PREPARATION_CRASHED', 'Input Preparation Crashed'),
        ('SKIPPED', 'Skipped'),
        ('TESTING', 'Testing'),
        ('REJECTED', 'Rejected'),
    ]

    user = models.ForeignKey(
        CodeforcesUser,
        on_delete=models.CASCADE,
        related_name='submissions'
    )

    # Codeforces submission ID (unique per submission)
    submission_id = models.BigIntegerField()

    # Contest / problem info
    contest_id = models.IntegerField(null=True, blank=True)
    problem_index = models.CharField(max_length=10)
    problem_name = models.CharField(max_length=255)
    problem_rating = models.IntegerField(null=True, blank=True)
    problem_tags = models.JSONField(default=list, blank=True)

    # Submission details
    creation_time_seconds = models.BigIntegerField()  # Unix timestamp from CF
    programming_language = models.CharField(max_length=100)
    verdict = models.CharField(
        max_length=50,
        choices=VERDICT_CHOICES,
        null=True,
        blank=True
    )
    time_consumed_millis = models.IntegerField(default=0)
    memory_consumed_bytes = models.BigIntegerField(default=0)

    # Store the full raw response for this submission
    raw_data = models.JSONField(default=dict)

    class Meta:
        # A user can't have two identical submission IDs
        unique_together = ('user', 'submission_id')
        ordering = ['-creation_time_seconds']

    def __str__(self):
        return f"{self.user.handle} | {self.problem_name} | {self.verdict}"


class UserStatusFetch(models.Model):
    """Audit log: records every time we fetched User.status for a user."""

    STATUS_CHOICES = [
        ('success', 'Success'),
        ('failed', 'Failed'),
    ]

    user = models.ForeignKey(
        CodeforcesUser,
        on_delete=models.CASCADE,
        related_name='fetches'
    )
    fetched_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    submissions_count = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ['-fetched_at']

    def __str__(self):
        return f"{self.user.handle} @ {self.fetched_at} [{self.status}]"
