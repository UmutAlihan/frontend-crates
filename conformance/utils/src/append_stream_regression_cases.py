# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append shared regression invariants to family-native stream-v1 inputs."""

import argparse
import difflib
from pathlib import Path

import yaml

from stream_regression_cases import REASONING_CASE, REASONING_CASES, SCALAR_CASE, SCALAR_CASES, STRING_CASE, STRING_CASES


def _fixture_path(inputs_root: Path, family: str, case_id: str) -> Path:
    if family in {"kimi_k3", "muse_glimmer"}:
        return inputs_root / family / "TOOLCALLING.streamv1.yaml"
    if case_id == REASONING_CASE:
        return inputs_root / family / "TOOLCALLING.streamv1.51.yaml"
    group = case_id.rsplit(".", 1)[0]
    return inputs_root / family / f"{group}.yaml"


def _append(inputs_root: Path, cases: dict[str, dict], case_id: str, *, replace_existing: bool = False) -> None:
    updates = {}
    for family, case in cases.items():
        path = _fixture_path(inputs_root, family, case_id)
        if path.exists():
            document = yaml.safe_load(path.read_text())
        else:
            template = path.with_name("TOOLCALLING.streamv1.50.yaml")
            document = {key: value for key, value in yaml.safe_load(template.read_text()).items() if key != "cases"}
            document["cases"] = {}
        if case_id in document["cases"]:
            existing = document["cases"][case_id]
            if existing == case:
                continue
            if not replace_existing:
                old_text = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True, width=4096).splitlines()
                new_text = yaml.safe_dump(case, sort_keys=False, allow_unicode=True, width=4096).splitlines()
                diff = "\n".join(
                    difflib.unified_diff(
                        old_text,
                        new_text,
                        fromfile=f"{path}:{case_id}:existing",
                        tofile=f"{path}:{case_id}:requested",
                        lineterm="",
                    )
                )
                raise ValueError(f"conflicting authored case {case_id} for {family}:\n{diff}")
        document["cases"][case_id] = case
        updates[path] = document

    for path, document in updates.items():
        path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=4096))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument(
        "--replace-case-id",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="Allow rewriting this exact authored case ID",
    )
    args = parser.parse_args()
    cases = {
        SCALAR_CASE: SCALAR_CASES,
        STRING_CASE: STRING_CASES,
        REASONING_CASE: REASONING_CASES,
    }
    unknown = sorted(set(args.replace_case_id) - cases.keys())
    if unknown:
        parser.error(f"unknown case IDs in --replace-case-id: {unknown}")
    for case_id, family_cases in cases.items():
        _append(args.inputs_root, family_cases, case_id, replace_existing=case_id in args.replace_case_id)


if __name__ == "__main__":
    main()
