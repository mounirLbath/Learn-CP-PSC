from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='CodeforcesUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('handle', models.CharField(max_length=100, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='UserSubmission',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('submission_id', models.BigIntegerField()),
                ('contest_id', models.IntegerField(blank=True, null=True)),
                ('problem_index', models.CharField(max_length=10)),
                ('problem_name', models.CharField(max_length=255)),
                ('problem_rating', models.IntegerField(blank=True, null=True)),
                ('problem_tags', models.JSONField(blank=True, default=list)),
                ('creation_time_seconds', models.BigIntegerField()),
                ('programming_language', models.CharField(max_length=100)),
                ('verdict', models.CharField(
                    blank=True,
                    choices=[
                        ('OK', 'Accepted'), ('WRONG_ANSWER', 'Wrong Answer'),
                        ('TIME_LIMIT_EXCEEDED', 'Time Limit Exceeded'),
                        ('MEMORY_LIMIT_EXCEEDED', 'Memory Limit Exceeded'),
                        ('RUNTIME_ERROR', 'Runtime Error'),
                        ('COMPILATION_ERROR', 'Compilation Error'),
                        ('CHALLENGED', 'Challenged'), ('FAILED', 'Failed'),
                        ('PARTIAL', 'Partial'), ('PRESENTATION_ERROR', 'Presentation Error'),
                        ('IDLENESS_LIMIT_EXCEEDED', 'Idleness Limit Exceeded'),
                        ('SECURITY_VIOLATED', 'Security Violated'), ('CRASHED', 'Crashed'),
                        ('INPUT_PREPARATION_CRASHED', 'Input Preparation Crashed'),
                        ('SKIPPED', 'Skipped'), ('TESTING', 'Testing'), ('REJECTED', 'Rejected'),
                    ],
                    max_length=50,
                    null=True,
                )),
                ('time_consumed_millis', models.IntegerField(default=0)),
                ('memory_consumed_bytes', models.BigIntegerField(default=0)),
                ('raw_data', models.JSONField(default=dict)),
                ('user', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='submissions',
                    to='cf_app.codeforcesuser',
                )),
            ],
            options={
                'ordering': ['-creation_time_seconds'],
                'unique_together': {('user', 'submission_id')},
            },
        ),
        migrations.CreateModel(
            name='UserStatusFetch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fetched_at', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(choices=[('success', 'Success'), ('failed', 'Failed')], max_length=10)),
                ('submissions_count', models.IntegerField(default=0)),
                ('error_message', models.TextField(blank=True)),
                ('user', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='fetches',
                    to='cf_app.codeforcesuser',
                )),
            ],
            options={
                'ordering': ['-fetched_at'],
            },
        ),
    ]
