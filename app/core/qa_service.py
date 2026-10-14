import json
import re
from typing import Dict, List, Optional, Tuple
from app.api.schemas import AskResponse, SourceItem
from app.core.llm import BaseLLMClient
from app.core.prompts import CORRECTION_PROMPT, SYSTEM_PROMPT, format_context_prompt
from app.core.retriever import Retriever
from app.core.vector_store import SearchResult
from app.utils.logging import setup_logger

logger = setup_logger("rag_assistant.qa_service")

UNAVAILABLE_MSG = "I couldn't find this information in the provided documents."


class QAService:
    """Orchestrates retrieve -> prompt -> answer -> ground -> verify pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: BaseLLMClient,
        default_top_k: int = 4,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.default_top_k = default_top_k

    def answer_question(
        self,
        question: str,
        session_id: str,
        top_k: Optional[int] = None,
    ) -> AskResponse:
        cleaned_question = question.strip()
        if not cleaned_question:
            return AskResponse(
                answer=UNAVAILABLE_MSG,
                answerable=False,
                sources=[],
            )

        # 1. Retrieve candidates above similarity threshold
        retrieved_chunks = self.retriever.retrieve(
            query=cleaned_question,
            session_id=session_id,
            top_k=top_k,
        )

        # 2. If nothing relevant found above threshold, skip LLM call
        if not retrieved_chunks:
            logger.info(f"No chunks exceeded threshold for query: '{cleaned_question}' - skipping LLM call")
            return AskResponse(
                answer=UNAVAILABLE_MSG,
                answerable=False,
                sources=[],
            )

        # 3. Format prompt with XML-delimited chunks
        prompt_text = format_context_prompt(cleaned_question, retrieved_chunks)
        chunk_map: Dict[str, SearchResult] = {c.chunk_id: c for c in retrieved_chunks}

        # 4. Invoke LLM with temperature 0.0
        parsed_data = self._invoke_and_parse(prompt_text, cleaned_question, chunk_map)

        # 5. Fallback if parsing or grounding failed
        if not parsed_data or not parsed_data.get("answerable", False):
            return AskResponse(
                answer=parsed_data.get("answer", UNAVAILABLE_MSG) if parsed_data else UNAVAILABLE_MSG,
                answerable=False,
                sources=[],
            )

        # 6. Verify citations and extract verbatim excerpts
        cited_ids = parsed_data.get("cited_chunk_ids", [])
        valid_sources = self._extract_verified_sources(
            cited_ids=cited_ids,
            chunk_map=chunk_map,
            question=cleaned_question,
            answer=parsed_data.get("answer", ""),
        )

        if not valid_sources:
            logger.warning("Answerable response did not contain any valid retrieved citations - falling back to unavailable")
            return AskResponse(
                answer=UNAVAILABLE_MSG,
                answerable=False,
                sources=[],
            )

        return AskResponse(
            answer=parsed_data.get("answer", "").strip(),
            answerable=True,
            sources=valid_sources,
        )

    def _invoke_and_parse(
        self,
        prompt: str,
        question: str,
        chunk_map: Dict[str, SearchResult],
    ) -> Optional[Dict]:
        """Invoke LLM and attempt to parse JSON response, with one retry on failure."""
        for attempt in range(2):
            current_prompt = prompt if attempt == 0 else f"{prompt}\n\n{CORRECTION_PROMPT}"
            try:
                raw_response = self.llm_client.generate(
                    prompt=current_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.0,
                )
                parsed = self._clean_and_parse_json(raw_response)
                if parsed and self._validate_structure(parsed, chunk_map):
                    return parsed
                logger.warning(f"Attempt {attempt + 1}: Model output failed validation: {raw_response[:200]}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: LLM generation error: {e}")

        return None

    def _clean_and_parse_json(self, raw_text: str) -> Optional[Dict]:
        text = raw_text.strip()
        # Remove Markdown code fence if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback regex search for json object
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    def _validate_structure(self, data: Dict, chunk_map: Dict[str, SearchResult]) -> bool:
        if not isinstance(data, dict):
            return False
        if "answer" not in data or "answerable" not in data:
            return False

        if not data["answerable"]:
            return True

        cited = data.get("cited_chunk_ids")
        if not isinstance(cited, list) or len(cited) == 0:
            return False

        # Ensure at least one cited ID exists in retrieved set
        valid_cites = [cid for cid in cited if cid in chunk_map]
        return len(valid_cites) > 0

    def _extract_verified_sources(
        self,
        cited_ids: List[str],
        chunk_map: Dict[str, SearchResult],
        question: str,
        answer: str,
    ) -> List[SourceItem]:
        sources: List[SourceItem] = []
        seen_chunks = set()

        for cid in cited_ids:
            if cid not in chunk_map or cid in seen_chunks:
                continue
            seen_chunks.add(cid)

            chunk = chunk_map[cid]
            excerpt = self._extract_verbatim_excerpt(chunk.text, question, answer)

            sources.append(
                SourceItem(
                    document=chunk.document_name,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    excerpt=excerpt,
                    score=chunk.score,
                )
            )

        # Sort sources by similarity score descending
        sources.sort(key=lambda s: s.score, reverse=True)
        return sources

    def _extract_verbatim_excerpt(self, full_text: str, question: str, answer: str) -> str:
        """Extract a real, verbatim sentence or continuous slice from chunk text."""
        # Split text into candidate sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]
        if not sentences:
            return full_text[:250].strip()

        query_words = set(re.findall(r"\w+", (question + " " + answer).lower())) - {
            "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "for", "of", "or"
        }

        best_sentence = sentences[0]
        max_matches = -1

        for sentence in sentences:
            sentence_words = set(re.findall(r"\w+", sentence.lower()))
            matches = len(query_words.intersection(sentence_words))
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence

        # Verify the sentence is an exact substring of full_text
        if best_sentence in full_text:
            return best_sentence[:350]
        return full_text[:250].strip()
