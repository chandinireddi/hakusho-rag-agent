import asyncio

from app.mcp_server import mcp


def test_mcp_exposes_read_only_tools():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert set(tools) == {"search_white_papers", "ask_white_papers"}
    assert all(t.annotations.read_only_hint for t in tools.values())
