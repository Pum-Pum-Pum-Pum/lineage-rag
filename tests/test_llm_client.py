from app.llm.client import generate_chat_completion, generate_chat_completion_with_usage
from app.llm.prompt_template import GroundedPrompt


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, choices: list[FakeChoice], usage: object | None = None) -> None:
        self.choices = choices
        self.usage = usage


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class FakeCompletionsAPI:
    def __init__(self, response: FakeChatResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, model: str, messages: list[dict]) -> FakeChatResponse:
        self.calls.append({"model": model, "messages": messages})
        return self.response


class FakeChatAPI:
    def __init__(self, response: FakeChatResponse) -> None:
        self.completions = FakeCompletionsAPI(response)


class FakeOpenAIClient:
    def __init__(self, response: FakeChatResponse) -> None:
        self.chat = FakeChatAPI(response)


def _prompt() -> GroundedPrompt:
    return GroundedPrompt(
        system_prompt="System rules",
        user_prompt="User question and evidence",
        citations=[],
    )


def test_generate_chat_completion_returns_content() -> None:
    fake_client = FakeOpenAIClient(
        FakeChatResponse([FakeChoice(" Grounded answer [C1]. ")], usage=FakeUsage())
    )

    answer = generate_chat_completion(
        prompt=_prompt(),
        model="test-model",
        client=fake_client,
    )

    assert answer == "Grounded answer [C1]."
    assert fake_client.chat.completions.calls[0]["model"] == "test-model"
    assert fake_client.chat.completions.calls[0]["messages"][0]["role"] == "system"
    assert fake_client.chat.completions.calls[0]["messages"][1]["role"] == "user"


def test_generate_chat_completion_with_usage_returns_content_and_usage() -> None:
    fake_client = FakeOpenAIClient(
        FakeChatResponse([FakeChoice("Grounded answer [C1].")], usage=FakeUsage())
    )

    result = generate_chat_completion_with_usage(
        prompt=_prompt(),
        model="test-model",
        client=fake_client,
    )

    assert result.content == "Grounded answer [C1]."
    assert result.usage.model == "test-model"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


def test_generate_chat_completion_with_usage_handles_missing_usage() -> None:
    fake_client = FakeOpenAIClient(FakeChatResponse([FakeChoice("Grounded answer [C1].")]))

    result = generate_chat_completion_with_usage(
        prompt=_prompt(),
        model="test-model",
        client=fake_client,
    )

    assert result.usage.model == "test-model"
    assert result.usage.prompt_tokens is None
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None


def test_generate_chat_completion_rejects_missing_choices() -> None:
    fake_client = FakeOpenAIClient(FakeChatResponse([]))

    try:
        generate_chat_completion(prompt=_prompt(), model="test-model", client=fake_client)
    except RuntimeError as exc:
        assert "did not contain any choices" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing choices")


def test_generate_chat_completion_rejects_empty_content() -> None:
    fake_client = FakeOpenAIClient(FakeChatResponse([FakeChoice("   ")]))

    try:
        generate_chat_completion(prompt=_prompt(), model="test-model", client=fake_client)
    except RuntimeError as exc:
        assert "content was empty" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for empty content")
