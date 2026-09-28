# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate safe additions and explicitly approved corrections to stream archives."""

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

import yaml


def case_sha256(case: object) -> str:
    """Hash a case's semantic value with stable JSON serialization."""
    payload = json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _documents(path: Path, root: PurePosixPath, *, capture_version: str | None) -> tuple[dict[str, dict], dict[str, str]]:
    """Read fixture documents and hash non-YAML files under one archive root."""
    found = {}
    preserved = {}
    members = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member_path in members
                or (member_path != root and root not in member_path.parents)
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unexpected stream archive member: {member.name}")
            members.add(member_path)
            if member.isdir():
                continue
            relative = member_path.relative_to(root)
            with archive.extractfile(member) as source:
                payload = source.read()
            if member_path.suffix == ".yaml":
                if len(relative.parts) != 2:
                    raise ValueError(f"unexpected stream archive member: {member.name}")
                document = yaml.safe_load(payload)
                expected_capture = {"dynamo_v2": capture_version} if capture_version is not None else None
                if (
                    not isinstance(document, dict)
                    or document.get("family") != relative.parts[0]
                    or document.get("mode") != "streamv1"
                    or document.get("cases") is None
                    or not isinstance(document.get("cases"), dict)
                    or (capture_version is not None and document.get("captured_with") != expected_capture)
                    or (capture_version is None and "captured_with" in document)
                ):
                    raise ValueError(f"invalid stream archive document: {member.name}")
                found[str(relative)] = document
            else:
                preserved[str(relative)] = hashlib.sha256(payload).hexdigest()
    return found, preserved


def _approved_case_additions(
    old_docs: dict[str, dict],
    new_docs: dict[str, dict],
    old_members: dict[str, str],
    new_members: dict[str, str],
    allowed_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]] | None:
    """Return additions and approved corrections when prior records are preserved."""
    if old_members != new_members or not old_docs.keys() <= new_docs.keys():
        return None
    added = set()
    replaced = set()
    for name, old_doc in old_docs.items():
        new_doc = new_docs[name]
        old_metadata = {key: value for key, value in old_doc.items() if key != "cases"}
        new_metadata = {key: value for key, value in new_doc.items() if key != "cases"}
        if old_metadata != new_metadata:
            return None
        old_cases = old_doc["cases"]
        new_cases = new_doc["cases"]
        for case_id, old_case in old_cases.items():
            if case_id not in new_cases:
                return None
            if new_cases[case_id] != old_case:
                ident = (new_doc["family"], case_id)
                if ident not in allowed_replacements:
                    return None
                replaced.add(ident)
        added.update((new_doc["family"], case_id) for case_id in new_cases.keys() - old_cases.keys())
    for name, new_doc in new_docs.items():
        if name not in old_docs:
            added.update((new_doc["family"], case_id) for case_id in new_doc["cases"])
    return added, replaced


def stream_input_case_ids(path: Path) -> set[tuple[str, str]]:
    """Return the family/case IDs in a packaged stream input archive."""
    docs, _members = _documents(path, PurePosixPath("toolcalling/fixtures-stream-v1/inputs"), capture_version=None)
    return {
        (document["family"], case_id)
        for document in docs.values()
        for case_id in document["cases"]
    }


def stream_input_changes_allowed(
    existing: Path,
    candidate: Path,
    allowed_case_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> set[tuple[str, str]] | None:
    """Return additions when prior inputs are unchanged or case IDs are approved for correction."""
    changes = stream_input_changes(existing, candidate, allowed_case_replacements)
    return None if changes is None else changes[0]


def stream_input_changes(
    existing: Path,
    candidate: Path,
    allowed_case_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]] | None:
    """Return additions and corrections when all other input data is unchanged."""
    root = PurePosixPath("toolcalling/fixtures-stream-v1/inputs")
    old_docs, old_members = _documents(existing, root, capture_version=None)
    new_docs, new_members = _documents(candidate, root, capture_version=None)
    return _approved_case_additions(old_docs, new_docs, old_members, new_members, allowed_case_replacements)


def stream_capture_case_ids(path: Path, capture_root: str) -> set[tuple[str, str]]:
    """Return family/case IDs in a validated Dynamo v2 stream archive."""
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or not capture_root.startswith("dynamo_v2-"):
        raise ValueError(f"invalid stream capture root: {capture_root}")
    capture_version = capture_root.removeprefix("dynamo_v2-")
    docs, _members = _documents(path, root, capture_version=capture_version)
    return {
        (document["family"], case_id)
        for document in docs.values()
        for case_id in document["cases"]
    }


def validate_capture_receipt(
    receipt: dict,
    inputs_path: Path,
    capture_path: Path,
    capture_root: str,
    required_cases: set[tuple[str, str]] | frozenset[tuple[str, str]],
) -> bool:
    """Confirm a receipt binds corrected inputs to the candidate capture results."""
    if receipt.get("format") != "dynamo-stream-capture-receipt-v1":
        return False
    captures = receipt.get("captures")
    if not isinstance(captures, dict):
        return False
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or not capture_root.startswith("dynamo_v2-"):
        raise ValueError(f"invalid stream capture root: {capture_root}")
    version = capture_root.removeprefix("dynamo_v2-")
    input_docs, _input_members = _documents(
        inputs_path, PurePosixPath("toolcalling/fixtures-stream-v1/inputs"), capture_version=None
    )
    capture_docs, _capture_members = _documents(capture_path, root, capture_version=version)
    input_cases = {
        (document["family"], case_id): document["cases"][case_id]
        for document in input_docs.values()
        for case_id in document["cases"]
    }
    captured_cases = {
        (document["family"], case_id): document["cases"][case_id]
        for document in capture_docs.values()
        for case_id in document["cases"]
    }
    receipt_cases = captures.get(capture_root)
    if not isinstance(receipt_cases, dict):
        return False
    for family, case_id in required_cases:
        relative = next(
            (name for name, document in input_docs.items() if document["family"] == family and case_id in document["cases"]),
            None,
        )
        capture_relative = next(
            (name for name, document in capture_docs.items() if document["family"] == family and case_id in document["cases"]),
            None,
        )
        if relative is None or capture_relative is None:
            return False
        if relative != capture_relative:
            return False
        entries = receipt_cases.get(relative)
        if not isinstance(entries, dict):
            return False
        entry = entries.get(case_id)
        if not isinstance(entry, dict):
            return False
        if (
            entry.get("input_sha256") != case_sha256(input_cases[(family, case_id)])
            or entry.get("result_sha256") != case_sha256(captured_cases[(family, case_id)])
        ):
            return False
    return True


def stream_capture_updates_allowed(
    existing: Path,
    candidate: Path,
    capture_root: str,
    allowed_case_additions: set[tuple[str, str]],
    allowed_case_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    """Allow input-approved additions and explicit corrections without changing archive provenance."""
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or not capture_root.startswith("dynamo_v2-"):
        raise ValueError(f"invalid stream capture root: {capture_root}")
    capture_version = capture_root.removeprefix("dynamo_v2-")
    old_docs, old_members = _documents(existing, root, capture_version=capture_version)
    new_docs, new_members = _documents(candidate, root, capture_version=capture_version)
    changes = _approved_case_additions(old_docs, new_docs, old_members, new_members, allowed_case_replacements)
    return changes is not None and changes[0] <= allowed_case_additions
