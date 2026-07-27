"""Chat rendering helpers used by SFT (completion split) and DPO (thinking suppression)."""

from train._common import render_prompt_completion, render_preference_to_text


class ChatMLTok:
    """Minimal ChatML tokenizer stand-in.

    Emits <think> on an assistant *generation prompt* only when thinking is enabled
    (enable_thinking is not False) — mirroring how Qwen templates behave — so tests can
    check that enable_thinking=False actually suppresses it.
    """

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=None, tools=None):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        s = "".join(parts)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if enable_thinking is not False:
                s += "<think>\n\n</think>\n"
        return s


class NoThinkTok(ChatMLTok):
    """A tokenizer whose template does NOT accept enable_thinking (raises TypeError)."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, tools=None):
        return super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
            enable_thinking=False, tools=tools)


def test_prompt_completion_single_turn():
    tok = ChatMLTok()
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    prompt, completion = render_prompt_completion(tok, msgs, None, enable_thinking=False)
    assert prompt.endswith("<|im_start|>assistant\n")
    assert completion == "A<|im_end|>\n"
    assert "<think>" not in prompt        # suppressed
    assert "Q" in prompt and "A" not in prompt   # answer is not leaked into the prompt


def test_prompt_completion_multi_turn_tool_use():
    tok = ChatMLTok()
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "CALL"},
        {"role": "tool", "content": "RESULT"},
        {"role": "assistant", "content": "FINAL"},
    ]
    prompt, completion = render_prompt_completion(tok, msgs, None, enable_thinking=False)
    # Prompt stops at the first assistant turn; the whole assistant-side trajectory is completion.
    assert "Q" in prompt and "CALL" not in prompt
    assert "CALL" in completion and "RESULT" in completion and "FINAL" in completion


def test_prompt_completion_no_assistant_returns_none():
    tok = ChatMLTok()
    msgs = [{"role": "user", "content": "Q"}]
    assert render_prompt_completion(tok, msgs, None, enable_thinking=False) is None


def test_prompt_completion_degrades_without_enable_thinking_kwarg():
    tok = NoThinkTok()
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    prompt, completion = render_prompt_completion(tok, msgs, None, enable_thinking=False)
    assert completion == "A<|im_end|>\n"


def test_preference_suppresses_thinking():
    tok = ChatMLTok()
    ex = {
        "prompt": [{"role": "user", "content": "Q"}],
        "chosen": [{"role": "assistant", "content": "GOOD"}],
        "rejected": [{"role": "assistant", "content": "BAD"}],
    }
    out = render_preference_to_text(tok, ex, enable_thinking=False)
    assert isinstance(out["prompt"], str)
    assert "<think>" not in out["prompt"]
    assert out["chosen"] == "GOOD<|im_end|>\n"
    assert out["rejected"] == "BAD<|im_end|>\n"


def test_preference_flat_strings_passthrough():
    tok = ChatMLTok()
    ex = {"prompt": "Q", "chosen": "GOOD", "rejected": "BAD"}
    assert render_preference_to_text(tok, ex, enable_thinking=False) == ex
