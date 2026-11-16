from app.retriever import HybridRetriever, reciprocal_rank_fusion


def test_rrf_rewards_documents_ranked_high_by_both_retrievers():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], k=60)
    assert {fused[0][0], fused[1][0]} == {1, 2}
    assert fused[-1][0] in {3, 4}


def test_bm25_finds_exact_term(retriever):
    hits = retriever.search("Rapidus 半導体")
    assert hits[0].chunk.doc_id == "meti"
    assert hits[0].bm25_rank == 1


def test_multi_query_fusion(retriever):
    hits = retriever.search(["モントリオール", "人口 減少"], k=2)
    assert {h.chunk.doc_id for h in hits} == {"env", "mhlw"}


def test_document_filter(retriever):
    hits = retriever.search("半導体", doc_ids={"env"})
    assert all(h.chunk.doc_id == "env" for h in hits)


def test_dense_retrieval_with_stub_embeddings(chunks, settings):
    def embed(texts):  # 2-d toy embedding: "semiconductor-ness" vs everything else
        return [[1.0, 0.0] if ("半導体" in t or "chip" in t) else [0.0, 1.0] for t in texts]

    r = HybridRetriever(chunks, settings, embed_fn=embed, fingerprint="t")
    assert r.dense_enabled
    hits = r.search("chip factory", mode="dense", k=1)  # English query, no shared tokens
    assert hits[0].chunk.doc_id == "meti"
    assert (settings.cache_dir / f"emb-{settings.embedding_model}-t.npy").exists()


def test_embedding_outage_falls_back_to_bm25(chunks, settings):
    calls = {"n": 0}

    def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] > 1:  # the first call builds the index, later calls (queries) fail
            raise TimeoutError("embedding API down")
        return [[1.0, 0.0] for _ in texts]

    r = HybridRetriever(chunks, settings, embed_fn=flaky_embed, fingerprint="outage")
    hits = r.search("Rapidus 半導体")  # must not raise
    assert hits[0].chunk.doc_id == "meti"
    assert all(h.dense_rank is None for h in hits)  # BM25 only
