# advanced retrieval methods: embedding -> reranker -> window expand
from .reranker import BGEReranker
from .query_rewriter import rewrite_query
from .bm25_retriever import BM25Retriever
from .rrf import rrf_fuse
from .mmr import mmr_select
reranker = BGEReranker()

def advanced_multi_query_similarity_search(llm, vectorstore, chunks, user_query, 
                                            k=8, window_expansion=False, rrf_top_n=10, mode='hybrid',
                                            enable_mmr=True,mmr_top_n=8):
    '''
    Multi-query RAG (Reason → Rewrite → Retrieval):

    1. Rewrite user query → multiple research sub-queries
    2. Each sub-query:
        a) embedding similarity search (recall)
        b) BGE reranker (precision)
        c) neighbor expansion (context preservation)
    3. Merge & deduplicate
    '''
    #  ① rewrite query for better recall
    sub_queries = rewrite_query(llm, user_query)
    print("\n🧠 Generated sub-queries:", sub_queries)

    bm25 = BM25Retriever(chunks)
    candidate_chunks = []

    for q in sub_queries:
        # similarity score: the smallest, the best as this is distance
        dense_results = [doc for doc, _ in vectorstore.similarity_search_with_score(q, k)]
        if mode == "dense":
            merged = dense_results
        elif mode == "bm25":
            merged = bm25.search(q, k)
        elif mode == "hybrid":
            bm25_results = bm25.search(q, k)
            # dense + bm25 → RRF (dedup)
            merged = rrf_fuse([dense_results, bm25_results])[:rrf_top_n]
        else:
            raise ValueError("mode must be 'dense', 'bm25', or 'hybrid'")

        # ② reranker (cross-encoder BGE large)
        rerank_top_n = max(3, int(rrf_top_n / 2))
        reranked = reranker.rerank(q, merged, top_n=rerank_top_n)
        candidate_chunks.extend(reranked)

    # ③ optional MMR diversity re-ranking
    docs = dedup_docs(candidate_chunks)

    if enable_mmr and len(docs) > mmr_top_n:
        doc_embeddings = vectorstore.embedding_function.embed_documents([d.page_content for d in docs])
        query_vec = vectorstore.embedding_function.embed_query(user_query)

        mmr_docs = mmr_select(query_vec, doc_embeddings, docs, top_k=mmr_top_n)

    # ④ neighbor expansion (context completeness) - no need with current chunking strategy
    if window_expansion:
        window_size = 1

        final_results, seen = [], set()
        index_map = {(c.metadata["source"], c.metadata["chunk_number"]): i for i, c in enumerate(chunks)}

        for ck in docs:
            idx = index_map.get((ck.metadata["source"], ck.metadata["chunk_number"]))
            if idx is None:
                continue

            for i in range(max(0, idx - window_size), min(len(chunks), idx + window_size + 1)):
                key = (chunks[i].metadata["source"], chunks[i].metadata["chunk_number"])
                if key not in seen:
                    final_results.append(chunks[i])
                    seen.add(key)
        return final_results

    return mmr_docs if enable_mmr else docs

def dedup_docs(docs):
    seen = set()
    out = []
    for doc in docs:
        key = (doc.metadata.get("source"),doc.metadata.get("chunk_number"))
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out





    #     # step 1: embedding similarity search
    #     vec_results = vectorstore.similarity_search_with_score(q, k) 
    #     coarse_top_chunks = [doc for doc, _ in vec_results]

    #     if not coarse_top_chunks:
    #         return []

    #     # step 2: BGE cross-encoder reranker
    #     reranked = reranker.rerank(q, coarse_top_chunks, top_n=top_n)

    #     # step 3: neighbor expansion
    #     for doc in reranked:
    #         key = (doc.metadata["source"], doc.metadata["chunk_number"])
    #         current_index = index_map.get(key)

    #         if current_index is None:
    #             continue

    #         start = max(0, current_index - window_size)
    #         end = min(len(chunks), current_index + window_size + 1)

    #         for i in range(start, end):
    #             ck = chunks[i]
    #             ck_key = (ck.metadata["source"], ck.metadata["chunk_number"])

    #             if ck_key not in seen:
    #                 final_results.append(ck)
    #                 seen.add(ck_key)

    # return final_results