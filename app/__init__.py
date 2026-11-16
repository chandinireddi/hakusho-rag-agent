"""White-paper RAG agent package."""

__version__ = "2.3.0"  # used by the API, MCP server and UI footer; must match pyproject.toml (tested)

# Use the operating system's certificate store (Windows/macOS) for HTTPS.
# Needed on corporate networks whose proxy re-signs TLS with a company root CA:
# certificates are still fully verified — just against the OS trust store.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # optional dependency
    pass
