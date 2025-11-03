import re
import ast
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from retrieval import retrieval
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Citation(BaseModel):
    document: str = Field(..., description="The document name including extension")
    page: int = Field(..., description="The page number from the PDF")

class LLMResponse(BaseModel):
    answer: str = Field(..., description="LLM answer text, no JSON or citations inside")
    citations: List[int] = Field(..., description="chunk ids cited in the answer")

def llm_model():
    llm = ChatOpenAI(
        temperature=0.2,
        model='gpt-4o',                  
        openai_api_key=OPENAI_API_KEY,
    )
    return llm

def generate_response(llm, vectorstore, chunks, query):
    retrieved_chunks = retrieval.advanced_multi_query_similarity_search(
                        llm = llm,
                        vectorstore=vectorstore, 
                        chunks=chunks, 
                        user_query=query, 
                        k=8, 
                        window_expansion=False, # no need for context expansion with current chunking strategy
                        rrf_top_n=10,
                        mode="hybrid", # mode: 'dense' | 'bm25' | 'hybrid'
                        enable_mmr=True,
                        mmr_top_n=8
                        )
    
    # assign chunk_id for tracking
    chunks_with_ids = []
    for idx, chunk in enumerate(retrieved_chunks):
        chunk.metadata["chunk_id"] = idx + 1  # 1-based
        chunks_with_ids.append(chunk)
    
    # build system prompt
    system_instruction = SystemMessage(content="""
        You are a Retrieval-Augmented Generation assistant.
        Use ONLY the provided text chunks to answer the question.
        If you don't find any relevant information in the context, do not answer the user's query.
        If there is even a small bit of relevancy between the user's query and the context, 
        please provide a detailed answer based on the instructions above.

        You MUST reply strictly in JSON, following this schema:

        {
        "answer": "detailed answer here",
        "citations": [chunk_id, chunk_id]
        }

        Rules:
        - DO NOT hallucinate chunk ids. Only use from provided chunks.
        - Only quote text that exists in chunks provided.
        - If unsure, return: {"answer": "No relevant information found.", "citations": []}
        """)

    # context messages (chunk text only, metadata excluded)
    context_messages = [
        HumanMessage(content=f"[chunk_id={chunk.metadata['chunk_id']}] {chunk.page_content}")
        for chunk in chunks_with_ids
    ]
  
    # user query
    user_message = HumanMessage(content=query)

    # call LLM
    raw_response = llm.invoke([system_instruction] + context_messages + [user_message]).content

    # parse JSON using pydantic
    response = parse_llm_response(raw_response)

    # map chunk_ids back to metadata
    id_to_chunk = {c.metadata["chunk_id"]: c for c in retrieved_chunks}

    mapped_citations = []
    for cited_id in response.citations:
        chunk = id_to_chunk.get(cited_id)
        if chunk:
            mapped_citations.append(
                    {
                        "document": chunk.metadata["source"],
                        "page": chunk.metadata["page"],
                        "chunk_id": cited_id,
                        "chunk_content": chunk.page_content
                    }
                )
    response.citations = mapped_citations
    return response

def parse_llm_response(raw_response: str) -> LLMResponse:
    # Clean accidental Markdown formatting like ```json
    if isinstance(raw_response, LLMResponse):
        return raw_response

    if not isinstance(raw_response, str):
        raw_response = str(raw_response)

    cleaned = raw_response.strip().lstrip("```json").rstrip("```").strip()

    try:
        parsed = json.loads(cleaned)
        validated = LLMResponse(**parsed)
        return validated
    except (json.JSONDecodeError, ValidationError) as e:
        print("LLM output format error:", e)
        raise e
