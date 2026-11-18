from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from api.celery_utils import decode_celery_field
from api.models import Scan, StateChoices, Task
from celery import states
from celery.utils.saferepr import saferepr
from django.test import override_settings
from django_celery_results.models import TaskResult
from tasks.jobs.orphan_recovery import (
    _SKIP_RECOVERY,
    _reconcile_task_results,
    _recovery_attempt_count,
    advisory_lock,
    is_worker_alive,
    reconcile_orphans,
    reenqueueable_tasks,
    revoke_task,
)


def _orphan_result(*, name, kwargs, worker, created_minutes_ago, status=states.STARTED):
    """Create a TaskResult mimicking an in-flight task, backdated past the grace."""
    tr = TaskResult.objects.create(
        task_id=str(uuid4()),
        status=status,
        task_name=name,
        worker=worker,
        task_kwargs=repr(kwargs),
        task_args=repr([]),
    )
    TaskResult.objects.filter(pk=tr.pk).update(
        date_created=datetime.now(tz=UTC) - timedelta(minutes=created_minutes_ago)
    )
    tr.refresh_from_db()
    return tr


class TestDecodeCeleryField:
    def test_decodes_strict_json(self):
        assert decode_celery_field('{"enabled": true, "scan_id": null}', {}) == {
            "enabled": True,
            "scan_id": None,
        }

    def test_decodes_single_encoded_repr(self):
        assert decode_celery_field("{'tenant_id': 'abc'}", {}) == {"tenant_id": "abc"}

    def test_decodes_double_encoded(self):
        import json

        stored = json.dumps(repr({"tenant_id": "abc", "scan_id": "s1"}))
        assert decode_celery_field(stored, {}) == {
            "tenant_id": "abc",
            "scan_id": "s1",
        }

    def test_python_words_inside_strings_are_preserved(self):
        stored = repr(
            {
                "enabled": True,
                "scan_id": None,
                "label": "True North",
                "note": "None",
            }
        )

        assert decode_celery_field(stored, {}) == {
            "enabled": True,
            "scan_id": None,
            "label": "True North",
            "note": "None",
        }

    def test_empty_returns_default(self):
        assert decode_celery_field(None, {}) == {}
        assert decode_celery_field("", []) == []
        assert decode_celery_field("null", {}) == {}
        assert decode_celery_field("None", []) == []

    def test_empty_validates_default(self):
        with pytest.raises(ValueError):
            decode_celery_field("", {"value": ...})

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            decode_celery_field("<<not a literal>>", {})

    @pytest.mark.parametrize(
        "value",
        (
            "{'value': ...}",
            "{'value': {1, 2}}",
            "{'value': b'bytes'}",
            '{"value": NaN}',
        ),
    )
    def test_non_json_values_raise(self, value):
        with pytest.raises(ValueError):
            decode_celery_field(value, {})

    def test_truncated_repr_raises(self):
        kwargs_repr = saferepr(
            {"finding_ids": [str(uuid4()) for _ in range(30)]}, maxlen=1024
        )
        assert "..." in kwargs_repr

        with pytest.raises(ValueError):
            decode_celery_field(kwargs_repr, {})


@pytest.mark.django_db
class TestReconcileTaskResults:
    def _patches(self, alive):
        """Patch worker liveness, revoke, and the task registry for re-enqueue."""
        mock_app = MagicMock()
        mock_task = MagicMock()
        mock_app.tasks.get.return_value = mock_task
        return (
            patch("tasks.jobs.orphan_recovery.is_worker_alive", return_value=alive),
            patch("tasks.jobs.orphan_recovery.revoke_task"),
            patch("tasks.jobs.orphan_recovery.current_app", mock_app),
            mock_task,
        )

    def test_recovers_non_scan_task(self, tenants_fixture):
        """A NON-scan task (tenant-deletion) left orphaned is re-enqueued too."""
        tenant = tenants_fixture[0]
        tr = _orphan_result(
            name="tenant-deletion",
            kwargs={"tenant_id": str(tenant.id)},
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=1),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["recovered"]
        tr.refresh_from_db()
        assert tr.status == states.REVOKED  # stale result cleared (no pending alert)
        mock_task.apply_async.assert_called_once()
        call = mock_task.apply_async.call_args.kwargs
        assert call["kwargs"] == {"tenant_id": str(tenant.id)}
        assert call["task_id"] != tr.task_id  # fresh task id

    def test_truncated_kwargs_are_not_reenqueued(self, tenants_fixture):
        tenant = tenants_fixture[0]
        tr = _orphan_result(
            name="tenant-deletion",
            kwargs={"tenant_id": str(tenant.id)},
            worker="dead@gone",
            created_minutes_ago=60,
        )
        tr.task_kwargs = saferepr(
            {"finding_ids": [str(uuid4()) for _ in range(30)]}, maxlen=1024
        )
        assert "..." in tr.task_kwargs
        tr.save(update_fields=["task_kwargs"])
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)

        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=1),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        mock_task.apply_async.assert_not_called()

    def test_wrong_kwargs_shape_is_not_reenqueued(self, tenants_fixture):
        tr = _orphan_result(
            name="tenant-deletion",
            kwargs={"tenant_id": str(tenants_fixture[0].id)},
            worker="dead@gone",
            created_minutes_ago=60,
        )
        tr.task_kwargs = "[]"
        tr.save(update_fields=["task_kwargs"])
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)

        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=1),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        mock_task.apply_async.assert_not_called()

    def test_external_integration_task_is_not_reenqueued_by_default(
        self, tenants_fixture
    ):
        """External side-effect tasks without proven idempotency stay terminal.

        integration-s3 rebuilds its upload from worker-local files that do not
        survive the crash, so re-enqueuing it would upload nothing.
        """
        tr = _orphan_result(
            name="integration-s3",
            kwargs={
                "tenant_id": str(tenants_fixture[0].id),
                "provider_id": str(uuid4()),
                "output_directory": "/tmp/gone",
            },
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=1),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        mock_task.apply_async.assert_not_called()

    @override_settings(TASK_RECOVERY_SUMMARIES_ENABLED=False)
    def test_disabled_group_task_is_not_reenqueued(self, tenants_fixture):
        """A task whose group feature flag is off stays terminal, not re-enqueued."""
        tr = _orphan_result(
            name="scan-summary",
            kwargs={
                "tenant_id": str(tenants_fixture[0].id),
                "scan_id": str(uuid4()),
            },
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=1),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        mock_task.apply_async.assert_not_called()

    @override_settings(TASK_RECOVERY_SUMMARIES_ENABLED=False)
    def test_disabled_group_task_does_not_consume_recovery_attempt(
        self, tenants_fixture
    ):
        """A disabled-group task is failed without incrementing its Valkey attempt
        counter, so re-enabling the group does not start it at the cap."""
        tr = _orphan_result(
            name="scan-summary",
            kwargs={"tenant_id": str(tenants_fixture[0].id), "scan_id": str(uuid4())},
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count") as mock_count,
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        mock_count.assert_not_called()

    @pytest.mark.parametrize(
        "task_name",
        [
            "attack-paths-scan-perform",
            "attack-paths-cleanup-stale-scans",
        ],
    )
    def test_scan_task_is_skipped_entirely(self, tenants_fixture, task_name):
        """Attack Paths scan tasks are excluded: the watchdog never touches them."""
        tr = _orphan_result(
            name=task_name,
            kwargs={
                "tenant_id": str(tenants_fixture[0].id),
                "scan_id": str(uuid4()),
            },
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with p_alive, p_revoke, p_app:
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id not in result["recovered"]
        assert tr.task_id not in result["failed"]
        assert tr.task_id not in result["skipped"]
        mock_task.apply_async.assert_not_called()

    def _orphaned_scan(self, tenant, provider, task_name, state):
        """A scan whose task a dead worker left in flight, plus the next queued scan."""
        kwargs = {"tenant_id": str(tenant.id), "provider_id": str(provider.id)}
        orphan = _orphan_result(
            name=task_name,
            kwargs=kwargs,
            worker="dead@gone",
            created_minutes_ago=60,
        )
        stuck_scan = Scan.objects.create(
            tenant_id=tenant.id,
            provider=provider,
            name="Stuck scan",
            trigger=Scan.TriggerChoices.MANUAL,
            state=state,
            task=Task.objects.create(
                id=orphan.task_id, task_runner_task=orphan, tenant_id=tenant.id
            ),
        )
        queued_result = TaskResult.objects.create(
            task_id=str(uuid4()), task_name="scan-perform", status="QUEUED"
        )
        queued_scan = Scan.objects.create(
            tenant_id=tenant.id,
            provider=provider,
            name="Queued scan",
            trigger=Scan.TriggerChoices.MANUAL,
            state=StateChoices.AVAILABLE,
            task=Task.objects.create(
                id=queued_result.task_id,
                task_runner_task=queued_result,
                tenant_id=tenant.id,
            ),
        )
        return orphan, stuck_scan, queued_scan

    @pytest.mark.parametrize(
        "task_name, state",
        [
            ("scan-perform", StateChoices.EXECUTING),
            ("scan-perform-scheduled", StateChoices.EXECUTING),
            ("scan-perform", StateChoices.AVAILABLE),
        ],
    )
    def test_orphaned_scan_is_closed_and_the_next_queued_scan_dispatched(
        self,
        tenants_fixture,
        aws_provider,
        django_capture_on_commit_callbacks,
        task_name,
        state,
    ):
        """An orphaned scan is never re-run: it is failed and the queue advances."""
        tenant = tenants_fixture[0]
        orphan, stuck_scan, queued_scan = self._orphaned_scan(
            tenant, aws_provider, task_name, state
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.tasks.perform_scan_task.apply_async") as mock_dispatch,
            django_capture_on_commit_callbacks(execute=True),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert orphan.task_id in result["failed"]
        assert orphan.task_id not in result["recovered"]
        mock_task.apply_async.assert_not_called()  # the orphan itself is not re-run
        orphan.refresh_from_db()
        assert orphan.status == states.REVOKED
        stuck_scan.refresh_from_db()
        assert stuck_scan.state == StateChoices.FAILED
        assert stuck_scan.completed_at is not None
        queued_scan.refresh_from_db()
        assert queued_scan.state == StateChoices.AVAILABLE
        assert queued_scan.task.task_runner_task.status == states.PENDING
        mock_dispatch.assert_called_once_with(
            kwargs={
                "tenant_id": str(tenant.id),
                "scan_id": str(queued_scan.id),
                "provider_id": str(aws_provider.id),
            },
            task_id=str(queued_scan.task_id),
        )

    @pytest.mark.parametrize("alive, dry_run", [(True, False), (False, True)])
    def test_scan_on_a_live_worker_or_in_a_dry_run_is_left_running(
        self,
        tenants_fixture,
        aws_provider,
        django_capture_on_commit_callbacks,
        alive,
        dry_run,
    ):
        """A scan still in flight, or a dry run, changes no scan and dispatches none."""
        tenant = tenants_fixture[0]
        orphan, stuck_scan, queued_scan = self._orphaned_scan(
            tenant, aws_provider, "scan-perform", StateChoices.EXECUTING
        )
        p_alive, p_revoke, p_app, _ = self._patches(alive=alive)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.tasks.perform_scan_task.apply_async") as mock_dispatch,
            django_capture_on_commit_callbacks(execute=True),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=dry_run
            )

        assert orphan.task_id not in result["failed"]
        orphan.refresh_from_db()
        assert orphan.status == states.STARTED
        stuck_scan.refresh_from_db()
        assert stuck_scan.state == StateChoices.EXECUTING
        queued_scan.refresh_from_db()
        assert queued_scan.task.task_runner_task.status == "QUEUED"
        mock_dispatch.assert_not_called()

    def test_orphaned_scan_without_a_provider_in_its_kwargs_closes_no_scan(
        self, tenants_fixture, aws_provider, django_capture_on_commit_callbacks
    ):
        """Without the tenant and provider the scan cannot be found: nothing is guessed."""
        tenant = tenants_fixture[0]
        orphan, stuck_scan, _ = self._orphaned_scan(
            tenant, aws_provider, "scan-perform", StateChoices.EXECUTING
        )
        TaskResult.objects.filter(pk=orphan.pk).update(
            task_kwargs=repr({"tenant_id": str(tenant.id)})
        )
        p_alive, p_revoke, p_app, _ = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.tasks.perform_scan_task.apply_async") as mock_dispatch,
            django_capture_on_commit_callbacks(execute=True),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert orphan.task_id in result["failed"]
        stuck_scan.refresh_from_db()
        assert stuck_scan.state == StateChoices.EXECUTING
        mock_dispatch.assert_not_called()

    def test_jira_integration_task_is_not_reenqueued(self, tenants_fixture):
        """integration-jira stays terminal: re-running it would create duplicate Jira
        issues, so an orphaned send is failed instead of re-enqueued."""
        tenant = tenants_fixture[0]
        kwargs = {
            "tenant_id": str(tenant.id),
            "integration_id": str(uuid4()),
            "project_key": "PROWLER",
            "issue_type": "Task",
            "finding_ids": [str(uuid4()), str(uuid4())],
        }
        tr = _orphan_result(
            name="integration-jira",
            kwargs=kwargs,
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=1),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        tr.refresh_from_db()
        assert tr.status == states.REVOKED  # stale result cleared (no pending alert)
        mock_task.apply_async.assert_not_called()

    def test_skips_live_worker(self, tenants_fixture):
        tr = _orphan_result(
            name="tenant-deletion",
            kwargs={"tenant_id": str(tenants_fixture[0].id)},
            worker="alive@host",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=True)
        with p_alive, p_revoke, p_app:
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["skipped"]
        mock_task.apply_async.assert_not_called()

    def test_skips_recently_created(self, tenants_fixture):
        tr = _orphan_result(
            name="tenant-deletion",
            kwargs={"tenant_id": str(tenants_fixture[0].id)},
            worker="dead@gone",
            created_minutes_ago=0,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with p_alive, p_revoke, p_app:
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        # too recent: excluded by the grace window (not even a candidate)
        assert tr.task_id not in result["recovered"]
        mock_task.apply_async.assert_not_called()

    def test_denylisted_task_failed_not_reenqueued(self, tenants_fixture):
        """A non-allowlisted task is failed, never blind re-run."""
        tr = _orphan_result(
            name="some-non-idempotent-task",
            kwargs={"tenant_id": str(tenants_fixture[0].id)},
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=1),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        tr.refresh_from_db()
        assert tr.status == states.REVOKED
        mock_task.apply_async.assert_not_called()

    def test_recovery_cap_marks_failed(self, tenants_fixture):
        """When the recovery counter exceeds the cap, the task is failed not re-run."""
        tr = _orphan_result(
            name="tenant-deletion",
            kwargs={"tenant_id": str(tenants_fixture[0].id)},
            worker="dead@gone",
            created_minutes_ago=60,
        )
        p_alive, p_revoke, p_app, mock_task = self._patches(alive=False)
        with (
            p_alive,
            p_revoke,
            p_app,
            patch("tasks.jobs.orphan_recovery._recovery_attempt_count", return_value=4),
        ):
            result = _reconcile_task_results(
                grace_minutes=2, max_attempts=3, window_hours=6, dry_run=False
            )

        assert tr.task_id in result["failed"]
        mock_task.apply_async.assert_not_called()


@pytest.mark.django_db
class TestOrphanRecoveryHelpers:
    def test_advisory_lock_acquires_and_releases(self):
        with advisory_lock() as acquired:
            assert acquired is True

    def test_is_worker_alive_true_when_responds(self):
        inspect = MagicMock()
        inspect.ping.return_value = {"w@h": {"ok": "pong"}}
        with patch(
            "tasks.jobs.orphan_recovery.current_app.control.inspect",
            return_value=inspect,
        ):
            assert is_worker_alive("w@h") is True

    def test_is_worker_alive_false_when_silent(self):
        inspect = MagicMock()
        inspect.ping.return_value = None
        with patch(
            "tasks.jobs.orphan_recovery.current_app.control.inspect",
            return_value=inspect,
        ):
            assert is_worker_alive("w@h") is False

    def test_revoke_task_terminates_with_sigterm_by_default(self):
        task_result = MagicMock(task_id="task-id")
        with patch(
            "tasks.jobs.orphan_recovery.current_app.control.revoke"
        ) as mock_revoke:
            revoke_task(task_result)

        mock_revoke.assert_called_once_with("task-id", terminate=True, signal="SIGTERM")

    def test_recovery_attempt_count_increments(self):
        # Unique signature so the Valkey counter starts fresh for this test.
        kwargs_repr = repr({"probe": str(uuid4())})
        redis_client = MagicMock()
        redis_client.incr.side_effect = [1, 2]
        with patch("redis.from_url", return_value=redis_client):
            assert _recovery_attempt_count("probe-task", kwargs_repr, 6) == 1
            assert _recovery_attempt_count("probe-task", kwargs_repr, 6) == 2


class TestRecoveryFeatureFlags:
    def test_attack_paths_tasks_are_excluded_from_generic_recovery(self):
        assert {
            "attack-paths-scan-perform",
            "attack-paths-cleanup-stale-scans",
        } <= _SKIP_RECOVERY

    def test_all_groups_enabled_by_default(self):
        tasks = reenqueueable_tasks()
        assert "scan-summary" in tasks
        assert {"provider-deletion", "tenant-deletion"} <= tasks

    @override_settings(TASK_RECOVERY_SUMMARIES_ENABLED=False)
    def test_summaries_group_flag_excludes_summary_tasks(self):
        tasks = reenqueueable_tasks()
        assert "scan-summary" not in tasks
        assert "scan-compliance-overviews" not in tasks
        assert "provider-deletion" in tasks

    @override_settings(TASK_RECOVERY_DELETIONS_ENABLED=False)
    def test_deletions_group_flag_excludes_deletion_tasks(self):
        tasks = reenqueueable_tasks()
        assert "provider-deletion" not in tasks
        assert "tenant-deletion" not in tasks
        assert "scan-summary" in tasks


@pytest.mark.django_db
class TestRecoveryMasterFlag:
    @override_settings(TASK_RECOVERY_ENABLED=False)
    def test_master_flag_disables_task_recovery(self):
        with patch(
            "tasks.jobs.orphan_recovery._reconcile_task_results"
        ) as mock_reconcile:
            result = reconcile_orphans(grace_minutes=2, max_attempts=3, dry_run=False)

        mock_reconcile.assert_not_called()
        assert result["acquired"] is True
        assert result["enabled"] is False
        assert "attack_paths" not in result

    @override_settings(TASK_RECOVERY_ENABLED=True)
    def test_master_flag_enabled_runs_task_recovery(self):
        with patch(
            "tasks.jobs.orphan_recovery._reconcile_task_results",
            return_value={"recovered": [], "failed": [], "skipped": []},
        ) as mock_reconcile:
            result = reconcile_orphans(grace_minutes=2, max_attempts=3, dry_run=False)

        mock_reconcile.assert_called_once()
        assert "attack_paths" not in result
