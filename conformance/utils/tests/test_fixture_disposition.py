# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_fixtures  # noqa: E402
import capture_stimulus  # noqa: E402
import fixture_disposition  # noqa: E402
import generate_conformance_table as table  # noqa: E402
import package_fixtures  # noqa: E402
import refresh_dynamo_captures  # noqa: E402
import append_stream_regression_cases  # noqa: E402
import stream_capture_archive  # noqa: E402
import unified_history  # noqa: E402

_TEST_STREAM_CAPTURE_VERSION = "99.0.0"
_TEST_STREAM_CAPTURE_ROOT = f"dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}"


def test_canonicalize_unified_inputs_rejects_duplicate_scenario_owners():
    records = {
        ("gemma4", "UNIFIED.1-1"): {"scenario": "text_only"},
        ("gemma4", "UNIFIED.9-9"): {"scenario": "text_only"},
    }

    with pytest.raises(ValueError, match="duplicate Unified scenario ownership"):
        fixture_disposition.canonicalize_unified_inputs(records)


def test_canonicalize_unified_inputs_allows_one_historical_alias_rename():
    scenario = "gemma4_guided_json_visible_call_prose_before_reasoning"
    records = {
        ("gemma4", "UNIFIED.31-29"): {"scenario": scenario, "input": "old"},
        ("gemma4", "UNIFIED.g4-1"): {"scenario": scenario, "input": "new"},
    }

    canonical, aliases = fixture_disposition.canonicalize_unified_inputs(records)

    ident = (
        "gemma4",
        fixture_disposition.canonical_unified_case_key(
            "gemma4", "UNIFIED.g4-1", scenario
        ),
    )
    assert canonical == {ident: records[("gemma4", "UNIFIED.g4-1")]}
    assert aliases[("gemma4", "UNIFIED.31-29")] == ident
    assert aliases[("gemma4", "UNIFIED.g4-1")] == ident


def test_checked_in_manifest_pins_unified_history_store():
    repo_root = SRC.parents[2]
    manifest = json.loads((repo_root / "conformance/fixtures-manifest.json").read_text())
    pinned = next(shard for shard in manifest["shards"] if shard.get("format") == "unified-history")

    digest, size = unified_history.store_digest(repo_root / "conformance/fixtures-unified-v2")

    assert pinned["sha256"] == digest
    assert pinned["size"] == size


def test_checked_in_manifest_validates_inactive_unified_evidence():
    repo_root = SRC.parents[2]
    manifest = json.loads((repo_root / "conformance/fixtures-manifest.json").read_text())

    inactive = fixture_disposition.verify_inactive_shards(
        manifest,
        repo_root / "conformance/fixtures",
    )

    assert all(path.startswith("unified/") for path in inactive)


def test_checked_in_manifest_tracks_inactive_unified_evidence():
    repo_root = SRC.parents[2]
    manifest = json.loads((repo_root / "conformance/fixtures-manifest.json").read_text())
    inactive_paths = sorted(fixture_disposition.inactive_shards(manifest))

    untracked = []
    for path in inactive_paths:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "--error-unmatch",
                "--",
                f"conformance/fixtures/{path}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            untracked.append(path)

    assert not untracked, f"manifest inactive shards must be tracked: {untracked}"


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    conf = tmp_path / "conformance"
    store = conf / "fixtures"
    false_path = "unified/dynamo_v2-0.6.0.tar.gz"
    path = store / false_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original mislabeled evidence")
    inactive = {"path": false_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size, "disposition": "quarantined", "reason": "wrong producer"}
    manifest = {"snapshot": "test", "shards": [], "inactive_shards": [inactive]}
    manifest_path = conf / "fixtures-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", store)
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", store)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: tmp_path / "cache")
    return conf, store, manifest, manifest_path


def _case(base, directory, key, record):
    path = base / directory / "gemma4" / f"{key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"family": "gemma4", "mode": "unified", "cases": {key: record}}))


@pytest.mark.parametrize("name", ["golden_spec", "golden_spec-1271640-0"])
def test_merge_shards_discards_generated_unified_oracle_artifacts(evidence, name):
    _conf, store, manifest, manifest_path = evidence
    shard_path = f"unified/{name}.tar.gz"
    path = store / shard_path
    path.write_bytes(b"generated oracle")
    manifest["shards"] = [
        {"path": shard_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
    ]
    manifest_path.write_text(json.dumps(manifest))

    assert package_fixtures.merge_shards(
        [], prune=False, fixtures_dir=store, manifest_path=manifest_path
    ) == []


def test_package_pins_unified_history_without_writing_an_archive(evidence, tmp_path, monkeypatch):
    _conf, _store, _manifest, _manifest_path = evidence
    history_root = tmp_path / "fixtures-unified-v2"
    family = {
        "input_document": {"family": "gemma4", "mode": "unified"},
        "golden_document": {"family": "gemma4", "mode": "unified"},
        "cases": {
            "text_only": {
                "lifecycle": "active",
                "scenario": "text_only",
                "description": "Visible text",
                "policy": ["test-policy"],
                "display_id": "UNIFIED.1-1",
                "historical_ids": [],
                "request": {
                    "input": "hello",
                    "init": {},
                    "finish_reason": "stop",
                    "tools": [],
                    "chunks": [],
                },
                "golden": {"assembled": []},
            }
        },
    }
    capture = {
        "provenance": {"status": "legacy", "captured_with": {"dynamo_v2": "0.1.0"}},
        "document": {},
        "changes": {},
        "metadata_changes": {},
        "document_overrides": {},
    }
    (history_root / "families/gemma4").mkdir(parents=True)
    (history_root / "families/gemma4/inputs_and_golden.yaml").write_text(
        unified_history.dump_yaml(family)
    )
    (history_root / "families/gemma4/dynamo_v2-0.1.0.yaml").write_text(
        unified_history.dump_yaml(capture)
    )
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history_root)
    monkeypatch.setattr(package_fixtures, "PER_SUBDIR_TREES", ("unified",))

    stage = tmp_path / "stage"
    _case(
        stage / "unified",
        "inputs",
        "UNIFIED.1-1",
        {
            "scenario": "text_only",
            "description": "Visible text",
            "policy": ["test-policy"],
            **family["cases"]["text_only"]["request"],
        },
    )
    _case(stage / "unified", "golden", "UNIFIED.1-1", {"assembled": []})
    monkeypatch.setattr(unified_history, "update_store_from_loose", lambda *_args, **_kwargs: [])
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    shards = package_fixtures.build_shards(stage, blobs)

    digest, size = unified_history.store_digest(history_root)
    assert shards == [{"path": "unified-history", "format": "unified-history", "sha256": digest, "size": size}]
    assert not list(blobs.rglob("*.tar.gz"))


def test_package_dry_run_does_not_update_unified_history(evidence, tmp_path, monkeypatch):
    _conf, _store, _manifest, _manifest_path = evidence
    history_root = tmp_path / "fixtures-unified-v2"
    (history_root / "families").mkdir(parents=True)
    (history_root / "families/gemma4").mkdir(parents=True)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history_root)
    monkeypatch.setattr(package_fixtures, "PER_SUBDIR_TREES", ("unified",))

    def mutate_history(
        store_root,
        _capture_root,
        *,
        complete_snapshot,
            required_capture_dirs,
    ):
        assert complete_snapshot is False
        assert required_capture_dirs == frozenset()
        path = store_root / "families/gemma4/dynamo_v2-0.1.0.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dry run must not write here")
        return [path]

    monkeypatch.setattr(unified_history, "update_store_from_loose", mutate_history)
    monkeypatch.setattr(unified_history, "store_digest", lambda _root: ("digest", 1))

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    package_fixtures.build_shards(tmp_path / "stage", blobs, dry_run=True)

    assert not (history_root / "families/gemma4/dynamo_v2-0.1.0.yaml").exists()


@pytest.mark.parametrize(
    ("current_roots", "complete_snapshot"),
    [
        ((), False),
        (("inputs", "golden"), True),
    ],
)
def test_build_shards_passes_exact_inactive_unified_capture_directories(
    tmp_path,
    monkeypatch,
    current_roots,
    complete_snapshot,
):
    stage = tmp_path / "stage"
    (stage / "unified").mkdir(parents=True)
    for root in current_roots:
        (stage / "unified" / root).mkdir()
    history = tmp_path / "history"
    history.mkdir()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    observed = {}
    inactive = {
        "unified/dynamo_v2-0.6.0.tar.gz": {},
        "unified/dynamo_v2-0.6.0.patch1.tar.gz": {},
        "unified/inputs+pr200.tar.gz": {},
        "toolcalling/fixtures.tar.gz": {},
    }

    def update_store(
        _store,
        _loose,
        *,
        complete_snapshot,
        required_capture_dirs,
    ):
        observed["complete_snapshot"] = complete_snapshot
        observed["required_capture_dirs"] = required_capture_dirs
        return []

    monkeypatch.setattr(package_fixtures, "PER_SUBDIR_TREES", ("unified",))
    monkeypatch.setattr(package_fixtures, "preserved_evidence", lambda: inactive)
    monkeypatch.setattr(unified_history, "update_store_from_loose", update_store)
    monkeypatch.setattr(unified_history, "store_digest", lambda _root: ("0" * 64, 0))

    package_fixtures.build_shards(stage, blobs, history_root=history)

    assert observed == {
        "complete_snapshot": complete_snapshot,
        "required_capture_dirs": frozenset(),
    }


@pytest.mark.parametrize("failure_stage", ["history", "archives", "manifest"])
def test_package_staging_failure_preserves_live_generation(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    conformance = tmp_path / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    manifest_path = conformance / "fixtures-manifest.json"
    fixtures.mkdir(parents=True)
    history.mkdir()
    (fixtures / "generation").write_text("old archives")
    (history / "generation").write_text("old history")
    manifest_path.write_text(
        json.dumps({"snapshot": "old", "shards": [], "inactive_shards": []}) + "\n"
    )
    before = {
        "fixtures": (fixtures / "generation").read_bytes(),
        "history": (history / "generation").read_bytes(),
        "manifest": manifest_path.read_bytes(),
    }

    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history)
    monkeypatch.setattr(package_fixtures, "stage_fixtures", lambda _source, _destination: None)

    def build_shards(_loose, _blobs, _prune, *, history_root):
        if failure_stage == "history":
            (history_root / "generation").write_text("partial history")
            raise OSError("injected failure after history mutation")
        return []

    def sync_store(
        _blobs,
        _shards,
        _dry_run,
        _prune,
        *,
        fixtures_dir,
        manifest_path,
        approved_stream_replacements=frozenset(),
        stream_capture_receipt=None,
    ):
        if failure_stage == "archives":
            generation = fixtures_dir / "generation"
            generation.unlink()
            generation.write_text("partial archives")
            raise OSError("injected failure after archive mutation")

    def validate_candidate(_manifest, _fixtures_dir, _history_dir):
        if failure_stage == "manifest":
            raise OSError("injected failure after manifest write")

    monkeypatch.setattr(package_fixtures, "build_shards", build_shards)
    monkeypatch.setattr(package_fixtures, "sync_store", sync_store)
    monkeypatch.setattr(package_fixtures, "_validate_candidate_package", validate_candidate)

    with pytest.raises(OSError, match="injected failure after"):
        package_fixtures.package_snapshot(
            "new",
            "new America/Los_Angeles",
            {},
            {},
            dry_run=False,
            prune=False,
        )

    assert (fixtures / "generation").read_bytes() == before["fixtures"]
    assert (history / "generation").read_bytes() == before["history"]
    assert manifest_path.read_bytes() == before["manifest"]


def test_package_snapshots_loose_inputs_after_acquiring_writer_lock(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    fixtures.mkdir(parents=True)
    history.mkdir()
    (conformance / "fixtures-manifest.json").write_text(
        json.dumps({"snapshot": "old", "shards": [], "inactive_shards": []}) + "\n"
    )
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history)
    monkeypatch.setattr(package_fixtures, "build_shards", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(package_fixtures, "sync_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(package_fixtures, "_validate_candidate_package", lambda *_args: None)
    monkeypatch.setattr(unified_history, "publish_paths_transactionally", lambda *_args, **_kwargs: None)

    first_staged = threading.Event()
    second_staged = threading.Event()
    release_first = threading.Event()

    def stage(_source, _destination):
        if threading.current_thread().name == "first-packager":
            first_staged.set()
            assert release_first.wait(timeout=5)
        else:
            second_staged.set()

    monkeypatch.setattr(package_fixtures, "stage_fixtures", stage)
    errors = []

    def package():
        try:
            package_fixtures.package_snapshot(
                "new",
                "new America/Los_Angeles",
                {},
                {},
                dry_run=False,
                prune=False,
            )
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=package, name="first-packager")
    second = threading.Thread(target=package, name="second-packager")
    first.start()
    assert first_staged.wait(timeout=5)
    second.start()
    assert not second_staged.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert errors == []
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_staged.is_set()


def test_extract_holds_generation_lock_through_shard_materialization(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    manifest_path = conformance / "fixtures-manifest.json"
    fixtures.mkdir(parents=True)
    history.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot": "old",
                "shards": [
                    {
                        "path": "toolcalling/a.tar.gz",
                        "sha256": "a" * 64,
                        "size": 1,
                    }
                ],
                "inactive_shards": [],
            }
        )
        + "\n"
    )
    cache = tmp_path / "cache"
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", history)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: cache)
    reader_at_shard = threading.Event()
    release_reader = threading.Event()
    writer_acquired = threading.Event()

    def shard_file(_shard):
        reader_at_shard.set()
        assert release_reader.wait(timeout=5)
        return fixtures / "toolcalling/a.tar.gz"

    def materialize_shard(
        _shard,
        _source,
        destination,
        *,
        derived_release_versions=None,
        verbose=False,
    ):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "materialized").write_text("old")

    monkeypatch.setattr(extract_fixtures, "shard_file", shard_file)
    monkeypatch.setattr(extract_fixtures, "materialize_shard", materialize_shard)
    monkeypatch.setattr(
        extract_fixtures,
        "publish_extracted_snapshot",
        lambda temporary, *_args, **_kwargs: temporary,
    )
    errors = []

    def read():
        try:
            extract_fixtures.extract_snapshot(
                SimpleNamespace(full_refresh=False, dry_run=False, info=False, verbose=False)
            )
        except Exception as error:
            errors.append(error)

    def write():
        try:
            with unified_history._store_mutation_lock(history):
                writer_acquired.set()
        except Exception as error:
            errors.append(error)

    reader = threading.Thread(target=read)
    writer = threading.Thread(target=write)
    reader.start()
    assert reader_at_shard.wait(timeout=5)
    writer.start()
    assert not writer_acquired.wait(timeout=0.2)
    release_reader.set()
    reader.join(timeout=5)
    writer.join(timeout=5)

    assert errors == []
    assert not reader.is_alive()
    assert not writer.is_alive()
    assert writer_acquired.is_set()


def test_extract_excludes_false_release_but_retains_real_patch(evidence, tmp_path, monkeypatch, capsys):
    conf, store, manifest, manifest_path = evidence
    staging = tmp_path / "staging"
    _case(staging / "unified", "dynamo_v2-0.6.0.patch3", "UNIFIED.gemma-1", {"assembled": []})
    rel = "unified/dynamo_v2-0.6.0.patch3"
    sha, size = package_fixtures._tar_dir(staging / rel, rel, store / f"{rel}.tar.gz")
    manifest["shards"] = [{"path": f"{rel}.tar.gz", "sha256": sha, "size": size}]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["extract_fixtures.py"])
    extract_fixtures.main()
    snapshot = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert not (snapshot / "unified/dynamo_v2-0.6.0").exists()
    assert (snapshot / rel / "gemma4/UNIFIED.gemma-1.yaml").is_file()
    assert fixture_disposition.inactive_fixture_dirs(snapshot / "unified") == {"dynamo_v2-0.6.0"}
    # A warm cache must still reject loss or replacement of the inactive evidence.
    (store / manifest["inactive_shards"][0]["path"]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="pinned bytes"):
        extract_fixtures.main()


def test_inactive_archive_does_not_hide_capture_from_active_unified_history(evidence, tmp_path):
    conf, store, manifest, manifest_path = evidence
    manifest["shards"] = [
        {"path": "unified-history", "format": "unified-history", "sha256": "0" * 64, "size": 0}
    ]
    manifest_path.write_text(json.dumps(manifest))

    assert fixture_disposition.inactive_fixture_dirs(store / "unified") == set()

    snapshot = tmp_path / "snapshot"
    (snapshot / "unified").mkdir(parents=True)
    (snapshot / ".fixtures-state.json").write_text(
        json.dumps(
            {
                "shards": {"unified-history": "0" * 64},
                "inactive_shards": manifest["inactive_shards"],
            }
        )
    )
    assert fixture_disposition.inactive_fixture_dirs(snapshot / "unified") == set()


def test_quarantined_evidence_cannot_be_reactivated_or_rebuilt(evidence, tmp_path):
    _conf, _store, manifest, _path = evidence
    shard = {key: manifest["inactive_shards"][0][key] for key in ("path", "sha256", "size")}
    with pytest.raises(ValueError, match="also active"):
        fixture_disposition.active_shards(manifest | {"shards": [shard]})
    with pytest.raises(ValueError, match="cannot activate"):
        package_fixtures.merge_shards([shard], prune=True)
    with pytest.raises(ValueError, match="cannot overwrite"):
        package_fixtures.sync_store(tmp_path, [shard], dry_run=False, prune=False)


@pytest.mark.parametrize("change", [
    {"path": "unified/another.tar.gz"}, {"sha256": "0" * 64}, {"size": 100},
    {"reason": "new assessment"}, {"disposition": "superseded"},
])
def test_every_inactive_identity_field_invalidates_cache(evidence, change):
    _conf, _store, manifest, _path = evidence
    old = manifest["inactive_shards"]
    changed = [old[0] | change]
    assert extract_fixtures.fixtures_identity([], old) != extract_fixtures.fixtures_identity([], changed)
    assert not extract_fixtures._state_matches({"shards": {}, "inactive_shards": old}, {}, changed)


def test_inactive_order_does_not_change_identity(evidence):
    _conf, _store, manifest, _path = evidence
    first = manifest["inactive_shards"][0]
    second = first | {"path": "unified/other.tar.gz"}
    assert extract_fixtures.fixtures_identity([], [first, second]) == extract_fixtures.fixtures_identity([], [second, first])


@pytest.mark.parametrize("change", [
    {"path": "../outside.tar.gz"}, {"path": "/absolute.tar.gz"}, {"sha256": "bad"},
    {"disposition": "active"}, {"reason": ""}, {"size": -1},
])
def test_disposition_rejects_unpinned_or_ambiguous_evidence(evidence, change):
    _conf, _store, manifest, _path = evidence
    with pytest.raises(ValueError):
        fixture_disposition.inactive_shards({"inactive_shards": [manifest["inactive_shards"][0] | change]})


def test_existing_versioned_archive_cannot_be_overwritten(evidence, tmp_path):
    _conf, store, _manifest, _path = evidence
    path = store / "unified/dynamo_v2-0.3.4.patch2.tar.gz"
    path.write_bytes(b"historical bytes")
    shard = {"path": str(path.relative_to(store)), "sha256": "0" * 64, "size": 1}
    with pytest.raises(ValueError, match="immutable"):
        package_fixtures.sync_store(tmp_path, [shard], dry_run=False, prune=False)
    assert path.read_bytes() == b"historical bytes"


def test_current_dynamo_stream_archive_allows_new_cases_only(tmp_path):
    def archive(path, cases):
        source = tmp_path / f"{path.stem}.yaml"
        source.write_text(yaml.safe_dump({
            "family": "glm47",
            "mode": "streamv1",
            "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION},
            "cases": cases,
        }))
        with tarfile.open(path, "w:gz") as output:
            output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/TOOLCALLING.streamv1.7.yaml")

    old = tmp_path / "old.tar.gz"
    appended = tmp_path / "appended.tar.gz"
    rewritten = tmp_path / "rewritten.tar.gz"
    archive(old, {"TOOLCALLING.streamv1.7.a": {"chunks": [{"expected": []}]}})
    archive(appended, {
        "TOOLCALLING.streamv1.7.a": {"chunks": [{"expected": []}]},
        "TOOLCALLING.streamv1.7.g": {"chunks": [{"expected": [{"complete": True}]}]},
    })
    archive(rewritten, {"TOOLCALLING.streamv1.7.a": {"chunks": [{"expected": [{"complete": True}]}]}})

    allowed = {("glm47", "TOOLCALLING.streamv1.7.g")}
    assert stream_capture_archive.stream_capture_updates_allowed(old, appended, _TEST_STREAM_CAPTURE_ROOT, allowed)
    assert not stream_capture_archive.stream_capture_updates_allowed(old, appended, _TEST_STREAM_CAPTURE_ROOT, set())
    assert not stream_capture_archive.stream_capture_updates_allowed(old, rewritten, _TEST_STREAM_CAPTURE_ROOT, allowed)


def test_current_dynamo_stream_archive_allows_only_explicit_case_replacements(tmp_path):
    def archive(path, value):
        source = tmp_path / f"{path.stem}.yaml"
        source.write_text(yaml.safe_dump({
            "family": "glm47",
            "mode": "streamv1",
            "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION},
            "cases": {"TOOLCALLING.streamv1.7.g": {"chunks": [{"expected": [value]}]}},
        }))
        with tarfile.open(path, "w:gz") as output:
            output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/TOOLCALLING.streamv1.7.yaml")

    old = tmp_path / "old.tar.gz"
    corrected = tmp_path / "corrected.tar.gz"
    archive(old, {"arguments": '{"nullable":null}'})
    archive(corrected, {"arguments": '{"count":42}'})

    assert not stream_capture_archive.stream_capture_updates_allowed(old, corrected, _TEST_STREAM_CAPTURE_ROOT, set())
    assert stream_capture_archive.stream_capture_updates_allowed(
        old,
        corrected,
        _TEST_STREAM_CAPTURE_ROOT,
        set(),
        {("glm47", "TOOLCALLING.streamv1.7.g")},
    )
    assert not stream_capture_archive.stream_capture_updates_allowed(
        old,
        corrected,
        _TEST_STREAM_CAPTURE_ROOT,
        set(),
        {("glm47", "TOOLCALLING.streamv1.7.h")},
    )


@pytest.mark.parametrize("version_suffix", ["", ".patch1"], ids=["release", "patch"])
def test_sync_store_accepts_stream_capture_additions(evidence, tmp_path, version_suffix):
    _conf, store, _manifest, manifest_path = evidence
    version = f"{_TEST_STREAM_CAPTURE_VERSION}{version_suffix}"
    relative = f"toolcalling/fixtures-stream-v1/dynamo_v2-{version}.tar.gz"
    destination = store / relative
    destination.parent.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    candidate = blobs / relative
    candidate.parent.mkdir(parents=True)
    source = tmp_path / "case.yaml"
    input_relative = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    old_input = store / input_relative
    candidate_input = blobs / input_relative
    old_input.parent.mkdir(parents=True, exist_ok=True)
    candidate_input.parent.mkdir(parents=True, exist_ok=True)
    input_document = {"family": "glm47", "mode": "streamv1", "cases": {"TOOLCALLING.streamv1.7.a": {"chunks": []}}}
    source.write_text(yaml.safe_dump(input_document))
    with tarfile.open(old_input, "w:gz") as archive:
        archive.add(source, arcname="toolcalling/fixtures-stream-v1/inputs/glm47/TOOLCALLING.streamv1.7.yaml")
    input_document["cases"]["TOOLCALLING.streamv1.7.g"] = {"chunks": []}
    source.write_text(yaml.safe_dump(input_document))
    with tarfile.open(candidate_input, "w:gz") as archive:
        archive.add(source, arcname="toolcalling/fixtures-stream-v1/inputs/glm47/TOOLCALLING.streamv1.7.yaml")
    member = f"toolcalling/fixtures-stream-v1/dynamo_v2-{version}/glm47/TOOLCALLING.streamv1.7.yaml"
    document = {"family": "glm47", "mode": "streamv1", "captured_with": {"dynamo_v2": version},
                "cases": {"TOOLCALLING.streamv1.7.a": {"chunks": [{"expected": []}]}}}
    source.write_text(yaml.safe_dump(document))
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname=member)
    document["cases"]["TOOLCALLING.streamv1.7.g"] = {"chunks": [{"expected": []}]}
    source.write_text(yaml.safe_dump(document))
    with tarfile.open(candidate, "w:gz") as archive:
        archive.add(source, arcname=member)
    shard = {"path": relative, "sha256": package_fixtures.sha256_file(candidate), "size": candidate.stat().st_size}

    input_shard = {"path": input_relative, "sha256": package_fixtures.sha256_file(candidate_input), "size": candidate_input.stat().st_size}
    package_fixtures.sync_store(blobs, [input_shard, shard], dry_run=False, prune=False, fixtures_dir=store, manifest_path=manifest_path)

    assert package_fixtures.sha256_file(destination) == shard["sha256"]
    assert package_fixtures.sha256_file(old_input) == input_shard["sha256"]


def test_sync_store_rejects_included_but_stale_capture_without_receipt(evidence, tmp_path):
    _conf, store, manifest, manifest_path = evidence
    input_relative = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    capture_paths = [
        f"toolcalling/fixtures-stream-v1/dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}.tar.gz",
        f"toolcalling/fixtures-stream-v1/dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}.patch1.tar.gz",
    ]
    case_id = "TOOLCALLING.streamv1.7.g"
    input_member = "toolcalling/fixtures-stream-v1/inputs/glm47/TOOLCALLING.streamv1.7.yaml"

    def write_archive(path, member, document):
        source = tmp_path / f"{path.name}.yaml"
        source.write_text(yaml.safe_dump(document))
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as archive:
            archive.add(source, arcname=member)

    old_input = store / input_relative
    write_archive(old_input, input_member, {
        "family": "glm47",
        "mode": "streamv1",
        "cases": {case_id: {"tools": [{"parameters": {"properties": {"count": {"type": ["integer", "null"]}}}}]}},
    })
    candidate_input = tmp_path / "blobs" / input_relative
    write_archive(candidate_input, input_member, {
        "family": "glm47",
        "mode": "streamv1",
        "cases": {case_id: {"tools": [{"parameters": {"properties": {"count": {"type": "integer"}}}}]}},
    })
    for path in capture_paths:
        capture_root = Path(path).name.removesuffix(".tar.gz")
        old_capture = store / path
        member = f"toolcalling/fixtures-stream-v1/{capture_root}/glm47/TOOLCALLING.streamv1.7.yaml"
        write_archive(old_capture, member, {
            "family": "glm47",
            "mode": "streamv1",
            "captured_with": {"dynamo_v2": capture_root.removeprefix("dynamo_v2-")},
            "cases": {case_id: {"chunks": [{"expected": [{"complete": True, "arguments": json.dumps({"nullable": None})}]}]}},
        })

    manifest["shards"] = [{"path": input_relative}, *[{"path": path} for path in capture_paths]]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    shards = []
    for path in [input_relative, *capture_paths]:
        candidate = tmp_path / "blobs" / path
        if path != input_relative:
            source = store / path
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(source.read_bytes())
        shards.append({
            "path": path,
            "sha256": package_fixtures.sha256_file(candidate),
            "size": candidate.stat().st_size,
        })

    with pytest.raises(ValueError, match="requires a matching capture receipt"):
        package_fixtures.sync_store(
            tmp_path / "blobs",
            shards,
            dry_run=False,
            prune=False,
            fixtures_dir=store,
            manifest_path=manifest_path,
            approved_stream_replacements={case_id},
        )


def test_stream_capture_receipt_binds_input_and_result_hashes(tmp_path):
    input_archive = tmp_path / "inputs.tar.gz"
    capture_archive = tmp_path / "capture.tar.gz"
    case_id = "TOOLCALLING.streamv1.7.g"
    input_case = {"tools": [{"parameters": {"properties": {"count": {"type": "integer"}}}}]}
    result_case = {"chunks": [{"expected": [{"complete": True, "arguments": '{"count":42}'}]}]}
    input_member = "toolcalling/fixtures-stream-v1/inputs/glm47/TOOLCALLING.streamv1.7.yaml"
    capture_member = f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/TOOLCALLING.streamv1.7.yaml"

    def write(path, member, document):
        source = tmp_path / f"{path.name}.yaml"
        source.write_text(yaml.safe_dump(document))
        with tarfile.open(path, "w:gz") as archive:
            archive.add(source, arcname=member)

    write(input_archive, input_member, {"family": "glm47", "mode": "streamv1", "cases": {case_id: input_case}})
    write(capture_archive, capture_member, {
        "family": "glm47",
        "mode": "streamv1",
        "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION},
        "cases": {case_id: result_case},
    })
    receipt = {
        "format": "dynamo-stream-capture-receipt-v1",
        "captures": {
            _TEST_STREAM_CAPTURE_ROOT: {
                "glm47/TOOLCALLING.streamv1.7.yaml": {
                    case_id: {
                        "input_sha256": stream_capture_archive.case_sha256(input_case),
                        "result_sha256": stream_capture_archive.case_sha256(result_case),
                    }
                }
            }
        },
    }

    assert stream_capture_archive.validate_capture_receipt(
        receipt, input_archive, capture_archive, _TEST_STREAM_CAPTURE_ROOT, {("glm47", case_id)}
    )
    assert not stream_capture_archive.validate_capture_receipt(
        receipt, input_archive, capture_archive, _TEST_STREAM_CAPTURE_ROOT, {("glm47", "TOOLCALLING.streamv1.7.h")}
    )


def test_stream_capture_receipt_hashes_the_recorder_input_snapshot(tmp_path, monkeypatch):
    tree = tmp_path / "fixtures-stream-v1"
    source = tree / "inputs/glm47/TOOLCALLING.streamv1.7.yaml"
    source.parent.mkdir(parents=True)
    case_id = "TOOLCALLING.streamv1.7.g"
    input_case = {"tools": [{"parameters": {"properties": {"count": {"type": "integer"}}}}]}
    source_payload = yaml.safe_dump({"family": "glm47", "mode": "streamv1", "cases": {case_id: input_case}}).encode()
    source.write_bytes(source_payload)
    recorder_inputs = []

    def record_snapshot(_crate, _binary, arguments):
        recorder_inputs.append(Path(arguments[0]).read_bytes())
        source.write_text(yaml.safe_dump({
            "family": "glm47",
            "mode": "streamv1",
            "cases": {case_id: {"tools": [{"parameters": {"properties": {"count": {"type": "string"}}}}]}},
        }))
        return json.dumps({case_id: [{"deltas": [{"complete": True, "arguments": '{"count":1}'}]}]})

    monkeypatch.setattr(refresh_dynamo_captures, "ensure_tree", lambda _name: tree)
    monkeypatch.setattr(refresh_dynamo_captures, "V2_FAMILIES", ["glm47"])
    monkeypatch.setattr(refresh_dynamo_captures, "run_bin", record_snapshot)
    receipt_path = tmp_path / "receipt.json"

    refresh_dynamo_captures.refresh_stream(_TEST_STREAM_CAPTURE_VERSION, receipt_path)

    receipt = json.loads(receipt_path.read_text())
    entry = receipt["captures"][_TEST_STREAM_CAPTURE_ROOT]["glm47/TOOLCALLING.streamv1.7.yaml"][case_id]
    assert recorder_inputs == [source_payload]
    assert entry["input_sha256"] == stream_capture_archive.case_sha256(input_case)


def test_approved_replacement_requires_an_actual_input_correction(evidence, tmp_path):
    _conf, store, manifest, manifest_path = evidence
    input_relative = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    capture_relative = f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}.tar.gz"
    case_id = "TOOLCALLING.streamv1.7.g"
    input_path = store / input_relative
    input_member = "toolcalling/fixtures-stream-v1/inputs/glm47/TOOLCALLING.streamv1.7.yaml"
    input_source = tmp_path / "input.yaml"
    input_source.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "cases": {case_id: {"tools": [{"parameters": {"properties": {"count": {"type": "integer"}}}}]}},
    }))
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(input_path, "w:gz") as archive:
        archive.add(input_source, arcname=input_member)

    old_capture = store / capture_relative
    capture_root = Path(capture_relative).name.removesuffix(".tar.gz")
    capture_member = f"toolcalling/fixtures-stream-v1/{capture_root}/glm47/TOOLCALLING.streamv1.7.yaml"
    capture_source = tmp_path / "capture.yaml"
    capture_source.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION},
        "cases": {case_id: {"chunks": [{"expected": [{"complete": True, "arguments": '{"count":1}'}]}]}},
    }))
    with tarfile.open(old_capture, "w:gz") as archive:
        archive.add(capture_source, arcname=capture_member)

    blobs = tmp_path / "blobs"
    candidate_input = blobs / input_relative
    candidate_input.parent.mkdir(parents=True)
    candidate_input.write_bytes(input_path.read_bytes())
    candidate_capture = blobs / capture_relative
    candidate_capture.parent.mkdir(parents=True, exist_ok=True)
    capture_source.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION},
        "cases": {case_id: {"chunks": [{"expected": [{"complete": True, "arguments": '{"count":2}'}]}]}},
    }))
    with tarfile.open(candidate_capture, "w:gz") as archive:
        archive.add(capture_source, arcname=capture_member)
    manifest["shards"] = [{"path": input_relative}, {"path": capture_relative}]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    shards = [
        {"path": path, "sha256": package_fixtures.sha256_file(blobs / path), "size": (blobs / path).stat().st_size}
        for path in (input_relative, capture_relative)
    ]

    with pytest.raises(ValueError, match="versioned capture is immutable"):
        package_fixtures.sync_store(
            blobs,
            shards,
            dry_run=False,
            prune=False,
            fixtures_dir=store,
            manifest_path=manifest_path,
            approved_stream_replacements={case_id},
        )


def test_stream_input_archive_allows_only_appended_cases(tmp_path):
    old = tmp_path / "old-inputs.tar.gz"
    candidate = tmp_path / "candidate-inputs.tar.gz"
    source = tmp_path / "input.yaml"
    original = {"family": "glm47", "mode": "streamv1", "cases": {"TOOLCALLING.streamv1.7.a": {"chunks": []}}}

    def archive(path, document):
        source.write_text(yaml.safe_dump(document))
        with tarfile.open(path, "w:gz") as output:
            output.add(source, arcname="toolcalling/fixtures-stream-v1/inputs/glm47/TOOLCALLING.streamv1.7.yaml")

    archive(old, original)
    appended = {**original, "cases": {**original["cases"], "TOOLCALLING.streamv1.7.g": {"chunks": []}}}
    archive(candidate, appended)
    assert stream_capture_archive.stream_input_changes_allowed(old, candidate) == {
        ("glm47", "TOOLCALLING.streamv1.7.g")
    }

    rewritten = {**original, "cases": {"TOOLCALLING.streamv1.7.a": {"chunks": [{"delta_text": "changed"}]}}}
    archive(candidate, rewritten)
    assert stream_capture_archive.stream_input_changes_allowed(old, candidate) is None
    assert stream_capture_archive.stream_input_changes_allowed(
        old,
        candidate,
        {("glm47", "TOOLCALLING.streamv1.7.a")},
    ) == set()
    assert stream_capture_archive.stream_input_changes(
        old,
        candidate,
        {("glm47", "TOOLCALLING.streamv1.7.a")},
    ) == (set(), {("glm47", "TOOLCALLING.streamv1.7.a")})
    assert stream_capture_archive.stream_input_changes_allowed(
        old,
        candidate,
        {("glm47", "TOOLCALLING.streamv1.7.g")},
    ) is None
    removed = {**original, "cases": {}}
    archive(candidate, removed)
    assert stream_capture_archive.stream_input_changes_allowed(
        old,
        candidate,
        {("glm47", "TOOLCALLING.streamv1.7.a")},
    ) is None
    metadata_changed = {**original, "model_label": "Changed"}
    archive(candidate, metadata_changed)
    assert stream_capture_archive.stream_input_changes_allowed(
        old,
        candidate,
        {("glm47", "TOOLCALLING.streamv1.7.a")},
    ) is None


def test_sync_store_requires_every_affected_active_capture_for_input_correction(evidence, tmp_path):
    _conf, store, manifest, manifest_path = evidence
    input_relative = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    capture_paths = [
        f"toolcalling/fixtures-stream-v1/dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}.tar.gz",
        f"toolcalling/fixtures-stream-v1/dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}.patch1.tar.gz",
    ]
    case_id = "TOOLCALLING.streamv1.7.g"
    input_member = "toolcalling/fixtures-stream-v1/inputs/glm47/TOOLCALLING.streamv1.7.yaml"

    def write_input(path, case):
        source = tmp_path / f"{path.stem}.yaml"
        source.write_text(yaml.safe_dump({"family": "glm47", "mode": "streamv1", "cases": {case_id: case}}))
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as archive:
            archive.add(source, arcname=input_member)

    def write_capture(path, capture_root, arguments):
        source = tmp_path / f"{capture_root}.yaml"
        source.write_text(yaml.safe_dump({
            "family": "glm47",
            "mode": "streamv1",
            "captured_with": {"dynamo_v2": capture_root.removeprefix("dynamo_v2-")},
            "cases": {case_id: {"chunks": [{"expected": [{"complete": True, "arguments": arguments}]}]}},
        }))
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as archive:
            archive.add(source, arcname=f"toolcalling/fixtures-stream-v1/{capture_root}/glm47/TOOLCALLING.streamv1.7.yaml")

    input_store = store / input_relative
    write_input(input_store, {"tools": [{"parameters": {"properties": {"count": {"type": ["integer", "null"]}}}}]})
    blobs = tmp_path / "blobs"
    candidate_input = blobs / input_relative
    write_input(candidate_input, {"tools": [{"parameters": {"properties": {"count": {"anyOf": [{"type": "integer"}]}}}}]})
    capture_roots = [
        f"dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}",
        f"dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}.patch1",
    ]
    for path, capture_root in zip(capture_paths, capture_roots):
        write_capture(store / path, capture_root, '{"nullable":null}')
    first_candidate = blobs / capture_paths[0]
    write_capture(first_candidate, capture_roots[0], '{"count":42}')

    manifest["shards"] = [{"path": input_relative}, *[{"path": path} for path in capture_paths]]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    input_shard = {
        "path": input_relative,
        "sha256": package_fixtures.sha256_file(candidate_input),
        "size": candidate_input.stat().st_size,
    }
    capture_shard = {
        "path": capture_paths[0],
        "sha256": package_fixtures.sha256_file(first_candidate),
        "size": first_candidate.stat().st_size,
    }
    input_case = {"tools": [{"parameters": {"properties": {"count": {"anyOf": [{"type": "integer"}]}}}}]}
    result_case = {"chunks": [{"expected": [{"complete": True, "arguments": '{"count":42}'}]}]}
    stream_capture_receipt = {
        "format": "dynamo-stream-capture-receipt-v1",
        "captures": {
            _TEST_STREAM_CAPTURE_ROOT: {
                "glm47/TOOLCALLING.streamv1.7.yaml": {
                    case_id: {
                        "input_sha256": stream_capture_archive.case_sha256(input_case),
                        "result_sha256": stream_capture_archive.case_sha256(result_case),
                    }
                }
            }
        },
    }

    with pytest.raises(ValueError, match="corrected stream input requires recapturing active archive"):
        package_fixtures.sync_store(
            blobs,
            [input_shard, capture_shard],
            dry_run=False,
            prune=False,
            fixtures_dir=store,
            manifest_path=manifest_path,
            approved_stream_replacements={case_id},
            stream_capture_receipt=stream_capture_receipt,
        )
def test_current_dynamo_stream_archive_rejects_wrong_capture_root(tmp_path):
    old = tmp_path / "old.tar.gz"
    candidate = tmp_path / "candidate.tar.gz"
    source = tmp_path / "case.yaml"
    source.write_text(yaml.safe_dump({"family": "glm47", "mode": "streamv1", "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION}, "cases": {}}))
    with tarfile.open(old, "w:gz") as output:
        output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/case.yaml")
    with tarfile.open(candidate, "w:gz") as output:
        output.add(source, arcname=f"toolcalling/fixtures-stream-v1/dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}.patch1/glm47/case.yaml")

    with pytest.raises(ValueError, match="unexpected stream archive member"):
        stream_capture_archive.stream_capture_updates_allowed(old, candidate, _TEST_STREAM_CAPTURE_ROOT, set())


def test_current_dynamo_stream_archive_allows_new_capture_document(tmp_path):
    old = tmp_path / "old.tar.gz"
    candidate = tmp_path / "candidate.tar.gz"
    source = tmp_path / "case.yaml"
    document = {"family": "glm47", "mode": "streamv1", "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION}, "cases": {}}
    source.write_text(yaml.safe_dump(document))
    with tarfile.open(old, "w:gz") as output:
        output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/case.yaml")
    source.write_text(yaml.safe_dump({**document, "cases": {"TOOLCALLING.streamv1.7.g": {"chunks": []}}}))
    with tarfile.open(candidate, "w:gz") as output:
        output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/case.yaml")
        output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/added.yaml")

    allowed = {("glm47", "TOOLCALLING.streamv1.7.g")}
    assert stream_capture_archive.stream_capture_updates_allowed(old, candidate, _TEST_STREAM_CAPTURE_ROOT, allowed)


def test_stream_regression_append_is_idempotent_and_preserves_other_cases(tmp_path):
    inputs = tmp_path / "inputs"
    family = inputs / "glm47"
    family.mkdir(parents=True)
    template = family / "TOOLCALLING.streamv1.50.yaml"
    template.write_text(yaml.safe_dump({"family": "glm47", "mode": "streamv1", "cases": {}}))
    case_id = "TOOLCALLING.streamv1.90.a"
    path = family / "TOOLCALLING.streamv1.90.yaml"
    prior_case = {"description": "authored prior case", "chunks": []}
    path.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "cases": {"TOOLCALLING.streamv1.90.b": prior_case},
    }))
    requested = {"description": "new case", "chunks": []}

    append_stream_regression_cases._append(inputs, {"glm47": requested}, case_id)
    first_write = path.read_bytes()
    append_stream_regression_cases._append(inputs, {"glm47": requested}, case_id)

    result = yaml.safe_load(path.read_text())
    assert path.read_bytes() == first_write
    assert result["cases"] == {
        "TOOLCALLING.streamv1.90.b": prior_case,
        case_id: requested,
    }


def test_stream_regression_append_rejects_conflicting_case_with_diff(tmp_path):
    inputs = tmp_path / "inputs"
    family = inputs / "glm47"
    family.mkdir(parents=True)
    path = family / "TOOLCALLING.streamv1.90.yaml"
    existing = {"description": "existing", "chunks": []}
    path.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "cases": {"TOOLCALLING.streamv1.90.a": existing},
    }))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="conflicting authored case") as error:
        append_stream_regression_cases._append(
            inputs,
            {"glm47": {"description": "replacement", "chunks": []}},
            "TOOLCALLING.streamv1.90.a",
        )

    assert "-description: existing" in str(error.value)
    assert "+description: replacement" in str(error.value)
    assert path.read_bytes() == before


def test_stream_regression_append_validates_all_families_before_writing(tmp_path):
    inputs = tmp_path / "inputs"
    case_id = "TOOLCALLING.streamv1.90.a"
    requested = {"description": "new", "chunks": []}
    paths = {}
    for family, existing in [
        ("glm47", None),
        ("deepseek_v4", {"description": "existing", "chunks": []}),
    ]:
        family_root = inputs / family
        family_root.mkdir(parents=True)
        path = family_root / "TOOLCALLING.streamv1.90.yaml"
        document = {"family": family, "mode": "streamv1", "cases": {}}
        if existing is not None:
            document["cases"][case_id] = existing
        path.write_text(yaml.safe_dump(document))
        paths[family] = (path, path.read_bytes())

    with pytest.raises(ValueError, match="conflicting authored case"):
        append_stream_regression_cases._append(
            inputs,
            {"glm47": requested, "deepseek_v4": requested},
            case_id,
        )

    assert {family: path.read_bytes() for family, (path, _) in paths.items()} == {
        family: original for family, (_, original) in paths.items()
    }


def test_stream_regression_append_replaces_only_explicit_case(tmp_path):
    inputs = tmp_path / "inputs"
    family = inputs / "glm47"
    family.mkdir(parents=True)
    path = family / "TOOLCALLING.streamv1.90.yaml"
    prior_case = {"description": "existing", "chunks": []}
    sibling_case = {"description": "preserved sibling", "chunks": []}
    path.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "cases": {
            "TOOLCALLING.streamv1.90.a": prior_case,
            "TOOLCALLING.streamv1.90.b": sibling_case,
        },
    }))
    requested = {"description": "approved replacement", "chunks": []}

    append_stream_regression_cases._append(
        inputs,
        {"glm47": requested},
        "TOOLCALLING.streamv1.90.a",
        replace_existing=True,
    )

    result = yaml.safe_load(path.read_text())
    assert result["cases"] == {
        "TOOLCALLING.streamv1.90.a": requested,
        "TOOLCALLING.streamv1.90.b": sibling_case,
    }


@pytest.mark.parametrize("member", [
    f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/../../escape.yaml",
    f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/case.yaml",
])
def test_current_dynamo_stream_archive_rejects_ambiguous_members(tmp_path, member):
    old = tmp_path / "old.tar.gz"
    candidate = tmp_path / "candidate.tar.gz"
    source = tmp_path / "case.yaml"
    source.write_text(yaml.safe_dump({"family": "glm47", "mode": "streamv1", "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION}, "cases": {}}))
    with tarfile.open(old, "w:gz") as output:
        output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/case.yaml")
    with tarfile.open(candidate, "w:gz") as output:
        output.add(source, arcname=f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/case.yaml")
        output.add(source, arcname=member)

    with pytest.raises(ValueError, match="unexpected stream archive member"):
        stream_capture_archive.stream_capture_updates_allowed(old, candidate, _TEST_STREAM_CAPTURE_ROOT, set())


def test_current_dynamo_stream_archive_rejects_symlink_members(tmp_path):
    old = tmp_path / "old.tar.gz"
    candidate = tmp_path / "candidate.tar.gz"
    source = tmp_path / "case.yaml"
    source.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "captured_with": {"dynamo_v2": _TEST_STREAM_CAPTURE_VERSION},
        "cases": {},
    }))
    member_name = f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/case.yaml"
    with tarfile.open(old, "w:gz") as output:
        output.add(source, arcname=member_name)
    with tarfile.open(candidate, "w:gz") as output:
        output.add(source, arcname=member_name)
        link = tarfile.TarInfo(f"toolcalling/fixtures-stream-v1/{_TEST_STREAM_CAPTURE_ROOT}/glm47/linked.yaml")
        link.type = tarfile.SYMTYPE
        link.linkname = member_name
        output.addfile(link)

    with pytest.raises(ValueError, match="unexpected stream archive member"):
        stream_capture_archive.stream_capture_updates_allowed(old, candidate, _TEST_STREAM_CAPTURE_ROOT, set())


@pytest.mark.parametrize("records", [["missing.yaml"], [], ["a.yaml", "a.yaml"], [1]])
def test_complete_snapshot_rejects_corrupt_member_index(records):
    with pytest.raises(ValueError, match="capture snapshot"):
        fixture_disposition.capture_snapshot_members(
            json.dumps({"schema_version": 1, "records": records}).encode(), ["a.yaml"],
        )


def test_renderer_uses_semantic_capture_directories(evidence, monkeypatch):
    conf, _store, _manifest, _manifest_path = evidence
    label = "0.6.1"
    base = f"dynamo_v2-{label}"
    stimulus = {"input": "latest", "tools": [], "chunks": [{"delta_text": "latest"}]}
    _case(conf / "unified", "inputs", "UNIFIED.1-1", stimulus)
    _case(conf / "unified", base, "UNIFIED.1-1", {
        "capture_input": capture_stimulus.capture_input(stimulus),
        "assembled": [{"kind": "text", "text": "captured"}],
    })
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda: label)
    cases, _caps, _versions = table._load_unified_fixtures(conf / "unified")
    assert cases[0]["dynamo"][0]["text"] == "captured"


def test_loose_reader_carries_a_prior_semantic_capture_to_current_release(tmp_path, monkeypatch):
    base = tmp_path / "unified"
    key = "UNIFIED.gemma-1"
    _case(base, "inputs", key, {"scenario": "gemma4_guided_json_visible_call_prose_before_reasoning", "chunks": []})
    _case(base, "golden", key, {"assembled": []})
    _case(base, "dynamo_v2-0.6.0", key, {"assembled": [{"kind": "text", "text": "captured"}]})
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda: "0.6.1")
    cases, _caps, versions = table._load_unified_fixtures(base)
    assert versions["dynamo_v2_all"] == ["0.6.0", "0.6.1"]
    assert cases[0]["dynamo"][0]["text"] == "captured"
    assert cases[0]["dynamo_by_ver"]["0.6.1"]["inherited_from"] == "0.6.0"


@pytest.mark.parametrize(("family", "old", "new"), [
    ("gemma4", "UNIFIED.31-29", "UNIFIED.gemma-1"),
    ("gemma4", "UNIFIED.31-30", "UNIFIED.gemma-2"),
    ("qwen3", "UNIFIED.31.a", "UNIFIED.31-1"),
    ("gemma4", "UNIFIED.31.x", "UNIFIED.34-6"),
    ("qwen3", "UNIFIED.31-29", "UNIFIED.31-29"),
    ("qwen3", "UNIFIED.30.m", "UNIFIED.30-13"),
    ("qwen3", "UNIFIED.1.a", "UNIFIED.1-1"),
    ("muse_glimmer", "UNIFIED.31-26", "UNIFIED.muse-1"),
    ("gemma4", "UNIFIED.31.y", "UNIFIED.31.y"),
])
def test_historical_aliases_do_not_reassign_other_ids(family, old, new):
    assert fixture_disposition.historical_unified_case_key(family, old) == new


def test_conflicting_capture_aliases_fail_without_touching_files(tmp_path, monkeypatch):
    _case(tmp_path, "inputs", "UNIFIED.gemma-1", {"scenario": "alias", "chunks": []})
    _case(tmp_path, "dynamo_v2-0.3.4", "UNIFIED.31-29", {"assembled": []})
    _case(tmp_path, "dynamo_v2-0.3.4", "UNIFIED.g4-1", {"assembled": [{"kind": "text", "text": "conflict"}]})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.yaml")}
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda: "0.3.4")
    with pytest.raises(ValueError, match="conflicting historical aliases"):
        table._load_unified_fixtures(tmp_path)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.yaml")} == before


def test_identical_capture_aliases_are_accepted_with_cached_records(tmp_path, monkeypatch):
    _case(tmp_path, "inputs", "UNIFIED.gemma-1", {"scenario": "alias", "chunks": []})
    record = {"assembled": [{"kind": "text", "text": "same"}]}
    _case(tmp_path, "dynamo_v2-0.3.4", "UNIFIED.31-29", record)
    _case(tmp_path, "dynamo_v2-0.3.4", "UNIFIED.g4-1", record)
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda: "0.3.4")

    cases, _captures, _versions = table._load_unified_fixtures(tmp_path)

    assert cases[0]["dynamo"][0]["text"] == "same"
