import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.core.chunking import TextChunker
from app.core.embeddings import create_embedding_client
from app.core.ingestion import DocumentIngestionService
from app.core.llm import create_llm_client
from app.core.qa_service import QAService
from app.core.retriever import Retriever
from app.core.vector_store import ChromaVectorStore


import uuid

def ingest_sample_documents(
    docs_dir: str,
    ingestion_service: DocumentIngestionService,
    embedding_client,
    vector_store: ChromaVectorStore,
    session_id: str,
) -> int:
    """Ingest sample documents into the evaluation vector store."""
    files = [f for f in os.listdir(docs_dir) if not f.startswith(".")]
    ingested_count = 0

    for filename in sorted(files):
        filepath = os.path.join(docs_dir, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "rb") as f:
            content = f.read()

        try:
            result = ingestion_service.process_document(filename, content)
            if not vector_store.has_document_by_hash(session_id, result.content_hash):
                embeddings = embedding_client.embed_batch([c.text for c in result.chunks])
                from app.api.schemas import DocumentInfo

                doc_info = DocumentInfo(
                    id=result.document_id,
                    filename=result.filename,
                    content_hash=result.content_hash,
                    chunk_count=len(result.chunks),
                    page_count=result.page_count,
                    file_size_bytes=result.file_size_bytes,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                vector_store.add_chunks(session_id, doc_info, result.chunks, embeddings)
                ingested_count += 1
                print(f"  [+] Ingested: {filename} ({len(result.chunks)} chunks)")
            else:
                print(f"  [=] Already present: {filename}")
        except Exception as e:
            print(f"  [!] Failed to ingest {filename}: {e}")

    return ingested_count


def run_evaluation(questions_path: str, results_path: str, docs_dir: str) -> Dict[str, Any]:
    print("=" * 70)
    print("RELIABLE RAG EVALUATION BENCHMARK RUNNER")
    print("=" * 70)

    settings = get_settings()

    # Create dedicated evaluation vector store directory and session
    eval_db_dir = "./data/chroma_eval_db"
    eval_session_id = str(uuid.uuid4())
    vector_store = ChromaVectorStore(
        persist_dir=eval_db_dir,
    )

    # Use embedding client
    embedding_client = create_embedding_client(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.llm_api_key,
    )

    chunker = TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    ingestion_service = DocumentIngestionService(chunker=chunker)

    print("\n1. Ingesting sample benchmark documents into evaluation session...")
    ingest_sample_documents(docs_dir, ingestion_service, embedding_client, vector_store, eval_session_id)

    retriever = Retriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        similarity_threshold=settings.similarity_threshold,
        default_top_k=settings.top_k,
    )

    llm_client = create_llm_client(
        provider=settings.llm_provider,
        model_name=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
    )

    qa_service = QAService(retriever=retriever, llm_client=llm_client, default_top_k=settings.top_k)

    print(f"\n2. Loading questions from: {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running evaluation on {len(questions)} test items...\n")

    results: List[Dict[str, Any]] = []
    total_questions = len(questions)
    answerable_total = 0
    answerable_correct = 0
    unanswerable_total = 0
    unanswerable_correct_refusals = 0
    total_expected_sources = 0
    matched_sources = 0
    total_keywords = 0
    matched_keywords = 0

    print(f"{'ID':<12} | {'Category':<15} | {'Exp':<5} | {'Pred':<5} | {'Src Match':<9} | {'Kw Match':<8} | Question")
    print("-" * 90)

    for q in questions:
        qid = q["id"]
        category = q["category"]
        text = q["question"]
        expected_ans = q["expected_answerable"]
        expected_docs = q.get("expected_source_documents", [])
        expected_kws = [kw.lower() for kw in q.get("expected_keywords", [])]

        t0 = time.time()
        resp = qa_service.answer_question(text, session_id=eval_session_id)
        elapsed = round(time.time() - t0, 3)

        pred_ans = resp.answerable
        pred_sources = [s.document for s in resp.sources]
        pred_text_lower = resp.answer.lower()

        is_decision_correct = (pred_ans == expected_ans)
        if expected_ans:
            answerable_total += 1
            if is_decision_correct:
                answerable_correct += 1
        else:
            unanswerable_total += 1
            if is_decision_correct:
                unanswerable_correct_refusals += 1

        # Check source document recall
        src_match = False
        if expected_docs:
            total_expected_sources += len(expected_docs)
            hits = sum(1 for ed in expected_docs if any(ed in ps for ps in pred_sources))
            matched_sources += hits
            src_match = (hits == len(expected_docs))
        else:
            src_match = (len(pred_sources) == 0)

        # Check keyword coverage
        kw_hits = 0
        if expected_kws:
            total_keywords += len(expected_kws)
            for kw in expected_kws:
                if kw in pred_text_lower:
                    kw_hits += 1
            matched_keywords += kw_hits
            kw_match_str = f"{kw_hits}/{len(expected_kws)}"
        else:
            kw_match_str = "N/A"

        src_str = "YES" if src_match else "NO"
        print(f"{qid:<12} | {category:<15} | {str(expected_ans):<5} | {str(pred_ans):<5} | {src_str:<9} | {kw_match_str:<8} | {text[:38]}...")

        results.append({
            "id": qid,
            "category": category,
            "question": text,
            "expected_answerable": expected_ans,
            "predicted_answerable": pred_ans,
            "decision_correct": is_decision_correct,
            "answer": resp.answer,
            "expected_sources": expected_docs,
            "retrieved_sources": pred_sources,
            "sources_matched": src_match,
            "expected_keywords": expected_kws,
            "keywords_matched": kw_hits,
            "latency_seconds": elapsed,
        })

    # Summary metrics
    overall_accuracy = (answerable_correct + unanswerable_correct_refusals) / total_questions if total_questions else 0.0
    refusal_precision = (unanswerable_correct_refusals / unanswerable_total) if unanswerable_total else 0.0
    source_recall = (matched_sources / total_expected_sources) if total_expected_sources else 1.0
    keyword_coverage = (matched_keywords / total_keywords) if total_keywords else 1.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_questions,
        "overall_accuracy": round(overall_accuracy, 4),
        "answerable_count": answerable_total,
        "answerable_correct": answerable_correct,
        "unanswerable_count": unanswerable_total,
        "correct_refusals": unanswerable_correct_refusals,
        "refusal_rate": round(refusal_precision, 4),
        "source_recall_rate": round(source_recall, 4),
        "keyword_coverage_rate": round(keyword_coverage, 4),
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.llm_provider,
    }

    print("\n" + "=" * 70)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"Total Test Questions:         {total_questions}")
    print(f"Overall Decision Accuracy:    {overall_accuracy * 100:.1f}% ({answerable_correct + unanswerable_correct_refusals}/{total_questions})")
    print(f"Correct Refusal Rate:         {refusal_precision * 100:.1f}% ({unanswerable_correct_refusals}/{unanswerable_total})")
    print(f"Source Document Recall:       {source_recall * 100:.1f}% ({matched_sources}/{total_expected_sources})")
    print(f"Fact / Keyword Coverage:      {keyword_coverage * 100:.1f}% ({matched_keywords}/{total_keywords})")
    print("=" * 70)

    output = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Evaluation report saved to: {results_path}\n")
    vector_store.delete_session(eval_session_id)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliable RAG Benchmark Evaluation")
    parser.add_argument("--questions", default="eval/questions.json", help="Path to questions.json")
    parser.add_argument("--results", default="eval/results.json", help="Path to output results.json")
    parser.add_argument("--docs", default="data/sample_docs", help="Path to sample documents directory")
    args = parser.parse_args()

    run_evaluation(args.questions, args.results, args.docs)
