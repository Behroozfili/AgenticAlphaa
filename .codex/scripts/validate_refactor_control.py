"""Validate refactor state, artifacts, transitions, and real Git changes.

Standard-library only. This command never runs product code, tests, agents,
migrations, network calls, or external writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Python 3.11+ is required (tomllib unavailable).") from exc


STAGE_ORDER = (
    "00", "01", "02", "03", "04A", "04B", "05", "06", "07", "08",
    "09", "10A", "10B", "11", "FINAL_QA", "FINAL_REVIEW",
)
STAGE_STATUSES = {
    "PENDING", "BLOCKED", "READY", "ACTIVE", "IMPLEMENTED",
    "QA_PASSED", "REVIEW_APPROVED", "CLOSED", "STALE",
}
PHASES = {"NOT_STARTED", "PRIMARY_DRAFT", "OBSERVABILITY", "PRIMARY_FINAL", "QA", "REVIEW", "CLOSED"}
MILESTONE_STATUSES = {"PENDING", "PASS", "FAIL", "BLOCKED"}
ATTEMPT_NAME = re.compile(
    r"^stage-(?P<stage>[0-9]{2}[AB]?)-rev-(?P<rev>[1-9][0-9]*)-attempt-"
    r"(?P<attempt>[1-9][0-9]*)-(?P<kind>draft-ready|implementation|observability|qa|review)\.json$"
)
HANDOFF_NAME = re.compile(r"^stage-(?P<stage>[0-9]{2}[AB]?)-rev-(?P<rev>[1-9][0-9]*)-handoff\.json$")
FINAL_NAME = re.compile(r"^final-(?P<kind>qa|review)-rev-(?P<rev>[1-9][0-9]*)\.json$")
CHANGE_NAME = re.compile(
    r"^change-request-stage-(?P<stage>[0-9]{2}[AB]?)-rev-(?P<rev>[1-9][0-9]*)-"
    r"attempt-(?P<attempt>[1-9][0-9]*)-[1-9][0-9]*\.json$"
)
MILESTONE_NAME = re.compile(
    r"^stage-(?P<stage>[0-9]{2}[AB]?)-rev-(?P<rev>[1-9][0-9]*)-attempt-"
    r"(?P<attempt>[1-9][0-9]*)-milestone-(?P<name>[A-Z][A-Z0-9_]*)\.json$"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def normalize_relative(raw: str) -> str:
    value = raw.replace("\\", "/").strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe/non-relative path: {raw!r}")
    return path.as_posix().lstrip("./")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _type_ok(value: Any, expected: str) -> bool:
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }
    return expected in checks and checks[expected]()


def validate_schema_instance(value: Any, schema: dict[str, Any], location: str = "$") -> list[str]:
    """Evaluate every JSON-Schema keyword used by artifact.schema.json."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, item) for item in choices):
            return [f"{location}: expected type {choices}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{location}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{location}: value {value!r} is not in enum")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{location}: string is shorter than minLength")
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"{location}: invalid date-time")
    if isinstance(value, int) and not isinstance(value, bool) and "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{location}: value is below minimum {schema['minimum']}")
    if isinstance(value, list):
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{location}: array items are not unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(validate_schema_instance(item, item_schema, f"{location}[{index}]"))
    if isinstance(value, dict):
        required = schema.get("required", [])
        for field in required:
            if field not in value:
                errors.append(f"{location}: missing required field {field!r}")
        properties = schema.get("properties", {})
        for field, child_schema in properties.items():
            if field in value:
                errors.extend(validate_schema_instance(value[field], child_schema, f"{location}.{field}"))
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                errors.append(f"{location}: additional properties not allowed: {sorted(extras)}")
    for child in schema.get("allOf", []):
        errors.extend(validate_schema_instance(value, child, location))
    if_schema = schema.get("if")
    then_schema = schema.get("then")
    if isinstance(if_schema, dict) and isinstance(then_schema, dict):
        if not validate_schema_instance(value, if_schema, location):
            errors.extend(validate_schema_instance(value, then_schema, location))
    return errors


def validate_state(root: Path, state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if state.get("schema_version") != "2.1":
        errors.append("state.schema_version must be 2.1")
    stages = state.get("stages")
    if not isinstance(stages, dict):
        return errors + ["state.stages must be an object"]
    if tuple(stages) != STAGE_ORDER:
        errors.append("state.stages must contain the canonical ordered stage set")
    active: list[str] = []
    stale_from_records: list[str] = []
    for stage_id, stage in stages.items():
        if not isinstance(stage, dict):
            errors.append(f"stage {stage_id} must be an object")
            continue
        status = stage.get("status")
        phase = stage.get("phase")
        if status not in STAGE_STATUSES:
            errors.append(f"stage {stage_id} has invalid status {status!r}")
        if phase not in PHASES:
            errors.append(f"stage {stage_id} has invalid phase {phase!r}")
        if status == "ACTIVE":
            active.append(stage_id)
        if status == "STALE":
            stale_from_records.append(stage_id)
        for dep in stage.get("dependencies", []):
            if dep not in stages:
                errors.append(f"stage {stage_id} has unknown dependency {dep}")
        milestones = stage.get("milestones")
        if not isinstance(milestones, dict):
            errors.append(f"stage {stage_id}.milestones must be an object")
            milestones = {}
        for name, milestone in milestones.items():
            if not isinstance(milestone, dict) or milestone.get("status") not in MILESTONE_STATUSES:
                errors.append(f"stage {stage_id} milestone {name} has invalid status object")
            if status == "CLOSED" and (milestone.get("status") != "PASS" or not milestone.get("artifact")):
                errors.append(f"closed stage {stage_id} milestone {name} lacks PASS evidence")
        if status == "CLOSED":
            if phase != "CLOSED" or not stage.get("accepted_attempt"):
                errors.append(f"closed stage {stage_id} lacks CLOSED phase/accepted attempt")
            required = {"implementation", "qa", "review", "handoff"}
            if stage.get("observability_required"):
                required |= {"draft_ready", "observability"}
            missing = required - set(stage.get("accepted_artifacts", {}))
            if missing:
                errors.append(f"closed stage {stage_id} missing accepted artifacts {sorted(missing)}")
    if len(active) > 1:
        errors.append(f"more than one ACTIVE stage: {active}")
    current = active[0] if active else None
    if state.get("active_stage") != current:
        errors.append("top-level active_stage does not match per-stage ACTIVE status")
    if sorted(state.get("stale_stages", [])) != sorted(stale_from_records):
        errors.append("top-level stale_stages does not match per-stage STALE records")
    if current:
        stage = stages[current]
        if not state.get("execution_enabled"):
            errors.append("execution_enabled=false while a stage is ACTIVE")
        for key, stage_key in (("stage_revision", "revision"), ("attempt", "attempt"), ("primary_agent", "primary_agent"), ("active_phase", "phase")):
            if state.get(key) != stage.get(stage_key):
                errors.append(f"top-level {key} does not match active stage")
        for dep in stage.get("dependencies", []):
            if stages[dep].get("status") != "CLOSED":
                errors.append(f"active stage {current} dependency {dep} is not CLOSED")
    else:
        if state.get("active_phase") is not None or state.get("primary_agent") is not None:
            errors.append("inactive state must have null active_phase and primary_agent")
        if state.get("stage_revision") != 0 or state.get("attempt") != 0:
            errors.append("inactive state must have top-level revision/attempt zero")
    for key in ("canonical_plan", "orchestration_spec"):
        path = root / str(state.get(key, ""))
        if not path.is_file():
            errors.append(f"missing {key}: {path}")
    for label, raw in state.get("contracts", {}).items():
        path = root / str(raw)
        if not path.is_file():
            errors.append(f"missing contract {label}: {path}")
    registered = {path.stem for path in (root / ".codex" / "agents").glob("*.toml")}
    for collection in ("write_locks", "permanent_write_owners"):
        for raw, owner in state.get(collection, {}).items():
            try:
                normalize_relative(raw)
            except ValueError as exc:
                errors.append(str(exc))
            if owner not in registered:
                errors.append(f"{collection} owner {owner!r} is not registered")
    schedule = state.get("protected_path_schedule", {})
    for raw, stage_owners in schedule.items():
        try:
            path = normalize_relative(raw)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(stage_owners, dict):
            errors.append(f"protected schedule for {path} must be an object")
            continue
        for stage_id, owner in stage_owners.items():
            if stage_id != "*" and stage_id not in stages:
                errors.append(f"protected schedule {path} has unknown stage {stage_id}")
            if owner not in registered:
                errors.append(f"protected schedule {path} has unknown owner {owner}")
    if current:
        for raw, owner in state.get("write_locks", {}).items():
            path = normalize_relative(raw)
            if path in schedule:
                expected = schedule[path].get(current, schedule[path].get("*"))
                if expected != owner:
                    errors.append(f"protected path {path} lock owner {owner!r} != scheduled {expected!r} for {current}")
    return errors


def validate_agents(root: Path) -> list[str]:
    errors: list[str] = []
    agent_dir = root / ".codex" / "agents"
    for path in sorted(agent_dir.glob("*.toml")):
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{path}: invalid TOML: {exc}")
            continue
        for field in ("name", "description", "developer_instructions"):
            if not isinstance(data.get(field), str) or not data[field].strip():
                errors.append(f"{path}: missing non-empty {field}")
        if data.get("name") != path.stem:
            errors.append(f"{path}: name must match filename")
    financial = (agent_dir / "financial.toml").read_text(encoding="utf-8")
    integration = (agent_dir / "integration.toml").read_text(encoding="utf-8")
    owned_line = "- `agents/executor_helpers/monthly_financial_features.py`"
    if owned_line not in financial:
        errors.append("Financial must own monthly_financial_features.py")
    if owned_line in integration:
        errors.append("Integration must not own monthly_financial_features.py")
    return errors


def validate_documents(root: Path) -> list[str]:
    errors: list[str] = []
    orchestration = (root / "codex_agents_refactor_orchestration_EN.md").read_text(encoding="utf-8")
    if "docs/implementation_plan.md` is canonical" in orchestration:
        errors.append("orchestration still names the missing plan as canonical")
    if "integration` owns the Stage 08 `api/routes/analyze.py`" in orchestration:
        errors.append("orchestration still assigns Stage 08 API route ownership")
    if '"artifacts": {' not in orchestration:
        errors.append("orchestration handoff example lacks the schema-required artifacts object")
    return errors


def validate_artifact(path: Path, data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors = [f"{path}: {item}" for item in validate_schema_instance(data, schema)]
    kind = data.get("artifact_type")
    match = ATTEMPT_NAME.match(path.name)
    if match:
        expected_kind = match.group("kind").replace("-", "_")
        if kind != expected_kind or data.get("stage") != match.group("stage") or data.get("stage_revision") != int(match.group("rev")) or data.get("attempt") != int(match.group("attempt")):
            errors.append(f"{path}: filename and payload identity differ")
    elif (match := HANDOFF_NAME.match(path.name)):
        if kind != "handoff" or data.get("stage") != match.group("stage") or data.get("stage_revision") != int(match.group("rev")):
            errors.append(f"{path}: handoff filename and payload identity differ")
    elif (match := FINAL_NAME.match(path.name)):
        expected_kind = f"final_{match.group('kind')}"
        expected_stage = f"FINAL_{match.group('kind').upper()}"
        if kind != expected_kind or data.get("stage") != expected_stage or data.get("stage_revision") != int(match.group("rev")):
            errors.append(f"{path}: final filename and payload identity differ")
    elif (match := CHANGE_NAME.match(path.name)):
        if kind != "change_request" or data.get("stage") != match.group("stage") or data.get("stage_revision") != int(match.group("rev")) or data.get("attempt") != int(match.group("attempt")):
            errors.append(f"{path}: change-request filename and payload identity differ")
    elif (match := MILESTONE_NAME.match(path.name)):
        if kind != "milestone" or data.get("stage") != match.group("stage") or data.get("stage_revision") != int(match.group("rev")) or data.get("attempt") != int(match.group("attempt")) or data.get("milestone_name") != match.group("name"):
            errors.append(f"{path}: milestone filename and payload identity differ")
    elif "templates" not in path.parts:
        errors.append(f"{path}: filename does not match artifact contract")
    return errors


def run_git(root: Path, *args: str) -> list[str]:
    proc = subprocess.run(["git", *args], cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return [normalize_relative(line) for line in proc.stdout.splitlines() if line.strip()]


def git_changed_files(root: Path, state: dict[str, Any]) -> set[str]:
    baseline = state.get("workspace_baseline", {}).get("git_ref", "HEAD")
    changed = set(run_git(root, "diff", "--name-only", "--relative", baseline))
    changed.update(run_git(root, "diff", "--cached", "--name-only", "--relative", baseline))
    untracked = set(run_git(root, "ls-files", "--others", "--exclude-standard"))
    ignored: set[str] = set()
    for item in state.get("workspace_baseline", {}).get("preexisting_untracked", []):
        path = normalize_relative(str(item.get("path", "")))
        full = root / path
        if path in untracked and full.is_file() and sha256_file(full) == str(item.get("sha256", "")).lower():
            ignored.add(path)
    changed.update(untracked - ignored)
    return changed


def ownership_map(state: dict[str, Any]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for item in state.get("lock_history", []):
        if item.get("status") == "ACCEPTED":
            owners[normalize_relative(item["path"])] = item["owner"]
    owners.update({normalize_relative(k): v for k, v in state.get("write_locks", {}).items()})
    owners.update({normalize_relative(k): v for k, v in state.get("permanent_write_owners", {}).items()})
    return owners


def validate_git_locks(root: Path, state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        changed = git_changed_files(root, state)
    except (RuntimeError, ValueError) as exc:
        return [f"unable to derive Git changes: {exc}"]
    owners = ownership_map(state)
    for path in sorted(changed):
        if path not in owners:
            errors.append(f"{path}: real Git change has no current/permanent/accepted lock owner")
    return errors


def validate_explicit_locks(state: dict[str, Any], agent: str, changed_files: list[str]) -> list[str]:
    owners = ownership_map(state)
    errors: list[str] = []
    for raw in changed_files:
        try:
            path = normalize_relative(raw)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if owners.get(path) != agent:
            errors.append(f"{path}: reported by {agent!r}, but attributable owner is {owners.get(path)!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-control", action="store_true")
    parser.add_argument("--check-locks", action="store_true")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--agent")
    parser.add_argument("--changed-files", nargs="*", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    try:
        state = load_json(root / ".codex" / "refactor-state.json")
        schema = load_json(root / ".codex" / "handoffs" / "schemas" / "artifact.schema.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CONTROL INVALID: {exc}", file=sys.stderr)
        return 1

    errors: list[str] = []
    if args.check_control:
        errors.extend(validate_state(root, state))
        errors.extend(validate_agents(root))
        errors.extend(validate_documents(root))
        for template in sorted((root / ".codex" / "handoffs" / "templates").glob("*.json")):
            try:
                errors.extend(validate_artifact(template, load_json(template), schema))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{template}: {exc}")
    if args.check_locks:
        errors.extend(validate_git_locks(root, state))
    for raw in args.artifact:
        path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        try:
            data = load_json(path)
            errors.extend(validate_artifact(path, data, schema))
            if data.get("changed_files") and data.get("agent"):
                errors.extend(validate_explicit_locks(state, data["agent"], data["changed_files"]))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if args.changed_files:
        if not args.agent:
            errors.append("--agent is required with --changed-files")
        else:
            errors.extend(validate_explicit_locks(state, args.agent, args.changed_files))

    if errors:
        print("CONTROL INVALID", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("CONTROL VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
