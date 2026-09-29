# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent value validator for the schema keywords authored by this corpus."""


def _resolve_local_ref(reference, root):
    assert reference.startswith("#/"), reference
    target = root
    for token in reference[2:].split("/"):
        target = target[token.replace("~1", "/").replace("~0", "~")]
    return target


def matches_schema(value, schema, root=None):
    root = schema if root is None else root
    if "$ref" in schema:
        target = _resolve_local_ref(schema["$ref"], root)
        siblings = {key: item for key, item in schema.items() if key != "$ref"}
        return matches_schema(value, target, root) and matches_schema(value, siblings, root)

    assert schema.keys() <= {"type", "properties", "items", "anyOf", "oneOf", "allOf", "const",
                             "enum", "nullable", "minLength", "minimum", "$defs"}, schema
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if schema.get("nullable") is True:
        kinds = kinds + ["null"]
    types = {"string": isinstance(value, str), "null": value is None,
             "number": type(value) in (int, float),
             "integer": type(value) is int or (type(value) is float and value.is_integer()),
             "boolean": type(value) is bool,
             "object": isinstance(value, dict), "array": isinstance(value, list)}
    # Reject unknown types even when another union member matches the value.
    assert all(isinstance(k, str) and k in types for k in kinds), schema
    if kinds and not any(types[k] for k in kinds):
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "minimum" in schema and (type(value) not in (int, float) or value < schema["minimum"]):
        return False
    if "anyOf" in schema and not any(matches_schema(value, branch, root) for branch in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(matches_schema(value, branch, root) for branch in schema["oneOf"]) != 1:
        return False
    if "allOf" in schema and not all(matches_schema(value, branch, root) for branch in schema["allOf"]):
        return False
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return all(matches_schema(item, properties[key], root) for key, item in value.items() if key in properties)
    if isinstance(value, list) and "items" in schema:
        return all(matches_schema(item, schema["items"], root) for item in value)
    return True
