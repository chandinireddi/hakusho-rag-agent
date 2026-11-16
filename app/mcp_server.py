"""MCP server: lets other AI agents (Claude Desktop, Cursor, LangGraph agents, …) use this
system as a tool. The agent, retriever and citations are the same as the web app.

    python -m app.mcp_server        # stdio transport (requires `pip install "mcp>=2"`)
"""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from app import __version__
from app.config import get_settings
from app.main import build_components
from app.schemas import AskRequest

mcp = MCPServer("hakusho-research", version=__version__)
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_components = None


def _c():
    global _components
    if _components is None:  # built on first use, so the server starts instantly
        _components = build_components(get_settings(), memory=False)
    return _components


@mcp.tool(annotations=READ_ONLY)
def search_white_papers(query: str, k: int = 5) -> list[dict]:
    """Search five 2023 Japanese government white papers (環境・厚労・総務・経産・文科).
    Returns passages with document title, ministry and PDF page. Japanese keywords work best."""
    hits = _c().retriever.search(query, k=max(1, min(k, 10)))
    return [
        {"title": h.chunk.title, "ministry": h.chunk.ministry, "page": h.chunk.page, "text": h.chunk.text} for h in hits
    ]


@mcp.tool(annotations=READ_ONLY)
def ask_white_papers(question: str) -> dict:
    """Answer a question (Japanese or English) from the white papers with page-level citations.
    `checks` is the 4-item answer inspection; status is 'answered' only if all checks pass,
    and 'insufficient_evidence' when the documents do not contain the answer."""
    from app.main import Run, to_response

    c = _c()
    req = AskRequest(question=question)
    run = Run.start(req)
    state = c.graph.invoke(run.state, run.config)
    res = to_response(req, run, state, c.settings)
    return res.model_dump(include={"answer", "status", "citations", "checks", "unsupported_claims", "numeric_issues"})


if __name__ == "__main__":
    mcp.run()
