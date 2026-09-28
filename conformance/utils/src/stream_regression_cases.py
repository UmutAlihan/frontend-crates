# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-native stream-v1 fixtures for shared parser regression invariants."""


SCALAR_CASE = "TOOLCALLING.streamv1.7.g"
STRING_CASE = "TOOLCALLING.streamv1.7.h"
REASONING_CASE = "TOOLCALLING.streamv1.51.a"


def scalar_tools():
    return [{"name": "inspect", "parameters": {"type": "object", "properties": {
        "count": {"anyOf": [{"type": "integer"}]},
        "ratio": {"type": ["number"]},
        "enabled": {"type": ["boolean"]},
    }}}]


def string_tools():
    return [{"name": "inspect", "parameters": {"type": "object", "properties": {
        "spaced": {"type": "string"},
        "blank": {"type": "string"},
        "empty": {"type": "string"},
    }}}]


WEATHER_TOOLS = [{"name": "get_weather", "parameters": {
    "type": "object", "properties": {"location": {"type": "string"}},
}}]


def finish(reason="tool_calls"):
    return {"delta_text": "", "finish_reason": reason}


def scalar_case(chunks):
    return {
        "description": "Composed scalar schemas preserve integer, number, and boolean values",
        "ref": "https://github.com/ai-dynamo/frontend-crates/pull/248",
        "tools": scalar_tools(),
        "chunks": chunks + [finish()],
    }


def string_case(chunks):
    return {
        "description": "Family-native string arguments preserve whitespace and empty strings",
        "ref": "https://github.com/ai-dynamo/frontend-crates/pull/247",
        "tools": string_tools(),
        "chunks": chunks + [finish()],
    }


def reasoning_case(chunks):
    return {
        "description": "Tool-only projection preserves caller-usable reasoning information around a tool call",
        "ref": "https://github.com/ai-dynamo/frontend-crates/pull/253",
        "tools": WEATHER_TOOLS,
        "chunks": chunks + [finish()],
    }


SCALAR_CASES = {
    "glm47": scalar_case([
        {"delta_text": "<tool_call>inspect<arg_key>count</arg_key><arg_value>42</arg_value><arg_key>ratio</arg_key><arg_value>1.25</arg_value>"},
        {"delta_text": "<arg_key>enabled</arg_key><arg_value>false</arg_value></tool_call>"},
    ]),
    "qwen3_coder": scalar_case([
        {"delta_text": "<tool_call><function=inspect><parameter=count>42</parameter><parameter=ratio>1.25</parameter>"},
        {"delta_text": "<parameter=enabled>false</parameter></function></tool_call>"},
    ]),
    "minimax_m2": scalar_case([
        {"delta_text": '<minimax:tool_call><invoke name="inspect"><parameter name="count">42</parameter><parameter name="ratio">1.25</parameter>'},
        {"delta_text": '<parameter name="enabled">false</parameter></invoke></minimax:tool_call>'},
    ]),
    "minimax_m3": scalar_case([
        {"delta_text": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="inspect">]<]minimax[>[<count>42]<]minimax[>[</count>]<]minimax[>[<ratio>1.25]<]minimax[>[</ratio>'},
        {"delta_text": ']<]minimax[>[<enabled>false]<]minimax[>[</enabled>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>'},
    ]),
}


STRING_CASES = {
    "deepseek_v4": string_case([
        {"delta_text": '<｜DSML｜tool_calls><｜DSML｜invoke name="inspect">'},
        {"delta_text": '<｜DSML｜parameter name="spaced" string="true">  café\n</｜DSML｜parameter>'},
        {"delta_text": '<｜DSML｜parameter name="blank" string="true">\t\r\n </｜DSML｜parameter>'},
        {"delta_text": '<｜DSML｜parameter name="empty" string="true"></｜DSML｜parameter>'},
        {"delta_text": '</｜DSML｜invoke></｜DSML｜tool_calls>'},
    ]),
    "glm47": string_case([
        {"delta_text": "<tool_call>inspect<arg_key>spaced</arg_key><arg_value>  café\n</arg_value>"},
        {"delta_text": "<arg_key>blank</arg_key><arg_value>\t\r\n </arg_value><arg_key>empty</arg_key><arg_value></arg_value>"},
        {"delta_text": "</tool_call>"},
    ]),
    "gemma4": string_case([
        {"delta_text": '<|tool_call>call:inspect{spaced:<|"|>  café\n<|"|>,blank:<|"|>\t\r\n <|"|>'},
        {"delta_text": ',empty:<|"|><|"|>}<tool_call|>'},
    ]),
    "kimi_k2": string_case([
        {"delta_text": "<|tool_calls_section_begin|><|tool_call_begin|>functions.inspect:0<|tool_call_argument_begin|>"},
        {"delta_text": '{"spaced":"  café\\n","blank":"\\t\\r\\n ","empty":""}'},
        {"delta_text": "<|tool_call_end|><|tool_calls_section_end|>"},
    ]),
    "kimi_k3": string_case([
        {"delta_text": '<|open|>tools<|sep|><|open|>call tool="inspect" index="1"<|sep|>'},
        {"delta_text": '<|open|>argument key="spaced" type="string"<|sep|>  café\n<|close|>argument<|sep|>'},
        {"delta_text": '<|open|>argument key="blank" type="string"<|sep|>\t\r\n <|close|>argument<|sep|>'},
        {"delta_text": '<|open|>argument key="empty" type="string"<|sep|><|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>'},
    ]),
    "minimax_m3": string_case([
        {"delta_text": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="inspect">]<]minimax[>[<spaced>  café\n]<]minimax[>[</spaced>'},
        {"delta_text": ']<]minimax[>[<blank>\t\r\n ]<]minimax[>[</blank>]<]minimax[>[<empty>]<]minimax[>[</empty>'},
        {"delta_text": ']<]minimax[>[</invoke>]<]minimax[>[</tool_call>'},
    ]),
    "muse_glimmer": string_case([
        {"delta_text": '<|start|>assistant to=inspect<|message|><atem:function_calls><atem:invoke name="inspect"><atem:parameter name="spaced">"  café\\n"</atem:parameter>'},
        {"delta_text": '<atem:parameter name="blank">"\\t\\r\\n "</atem:parameter><atem:parameter name="empty">""</atem:parameter>'},
        {"delta_text": '</atem:invoke></atem:function_calls><|eom|>'},
    ]),
}


REASONING_CASES = {
    "deepseek_v4": reasoning_case([
        {"delta_text": "before<think>reason</think>"},
        {"delta_text": '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'},
        {"delta_text": '<｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>'},
        {"delta_text": '</｜DSML｜invoke></｜DSML｜tool_calls>after'},
    ]),
    "kimi_k3": reasoning_case([
        {"delta_text": "<|open|>think<|sep|>reason<|close|>think<|sep|>"},
        {"delta_text": '<|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="location" type="string"<|sep|>Paris<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>'},
        {"delta_text": "after"},
    ]),
    "muse_glimmer": reasoning_case([
        {"delta_text": "<|start|>assistant to=self<|message|>reason<|eom|>"},
        {"delta_text": '<|start|>assistant to=get_weather<|message|><atem:function_calls><atem:invoke name="get_weather"><atem:parameter name="location">Paris</atem:parameter></atem:invoke></atem:function_calls><|eom|>after'},
    ]),
}
