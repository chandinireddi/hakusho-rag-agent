"""Shared fixtures (loaded by pytest via `-p tests.congest` in pyproject.toml)."""

from collections import defaultdict, deque

import pytest

from app.config import Settings
from app.ingest import Chunk
from app.retriever import HybridRetriever


class FakeLLM:
    """Stands in for ChatOpenAI. Queue the structured output each schema should return;
    every call is recorded so tests can assert what the agent did (or did not) ask."""

    def __init__(self):
        self._outputs: dict[str, deque] = defaultdict(deque)
        self.calls: list[str] = []
        self.messages: list[tuple[str, list]] = []

    def queue(self, schema, *outputs):
        self._outputs[schema.__name__].extend(outputs)

    def with_structured_output(self, schema, **_):
        llm = self

        class _Runnable:
            def invoke(self, messages, config=None):
                llm.calls.append(schema.__name__)
                llm.messages.append((schema.__name__, messages))
                return llm._outputs[schema.__name__].popleft()

        return _Runnable()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "documents").mkdir()
    return Settings(
        _env_file=None,  # never read the developer's real .env in tests
        openai_api_key="",
        documents_dir=tmp_path / "documents",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def chunks():
    return [
        Chunk("env:p8:c0", "env", "環境白書", "環境省", 8,
              "2022年12月にカナダ・モントリオールで生物多様性条約COP15第二部が開催された。"),
        Chunk("mhlw:p3:c0", "mhlw", "厚生労働白書", "厚生労働省", 3,
              "日本の人口は2008年をピークに減少に転じた。"),
        Chunk("meti:p17:c0", "meti", "通商白書", "経済産業省", 17,
              "Rapidusは北海道千歳市に次世代半導体の製造拠点を整備している。"),
        Chunk("soumu:p6:c0", "soumu", "情報通信白書", "総務省", 6,
              "SNSではエコーチェンバー現象により、自分と似た意見に触れやすくなる。"),
    ]  # fmt: skip


@pytest.fixture
def retriever(chunks, settings):
    return HybridRetriever(chunks, settings)  # no embed_fn → BM25 only, no network
