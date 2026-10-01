"""Controlled checks; these fixtures are not real embeddings or LLM answers.

Run: python -m unittest -v test_guardrails.py
"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import my_rag as rag


def chunk(distance, source="notes.txt", paragraph=1, text="A course note."):
    return {
        "id": f"{source}::paragraph-{paragraph}", "text": text,
        "source": source, "paragraph": paragraph, "distance": distance,
    }


class FixtureCollection:
    """Match Chroma's result shape with explicit distances; no embeddings."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_query = None

    def count(self):
        return len(self.chunks)

    def query(self, **kwargs):
        self.last_query = kwargs
        chunks = self.chunks[:kwargs["n_results"]]
        return {
            "ids": [[c["id"] for c in chunks]],
            "documents": [[c["text"] for c in chunks]],
            "metadatas": [[{"source": c["source"], "paragraph": c["paragraph"]} for c in chunks]],
            "distances": [[c["distance"] for c in chunks]],
        }


class GuardrailTests(unittest.TestCase):
    def ask(self, collection, question="Explain this.", **kwargs):
        with redirect_stdout(io.StringIO()):
            return rag.answer_question(collection, question, **kwargs)

    def test_strict_filter_and_prompt_exclusion(self):
        collection = FixtureCollection([
            chunk(0.2, paragraph=1, text="ACCEPTED_TEXT"),
            chunk(1.0, paragraph=2, text="EXCLUDED_AT_BOUNDARY"),
            chunk(1.2, paragraph=3, text="EXCLUDED_ABOVE_BOUNDARY"),
        ])
        with patch.object(rag, "generate_answer", return_value=(
            "An answer. [notes.txt:paragraph-1]", False,
        )) as generate:
            response = self.ask(collection)
        prompt = rag.build_rag_prompt("Explain this.", generate.call_args.args[1])
        self.assertIn("ACCEPTED_TEXT", prompt)
        self.assertNotIn("EXCLUDED_AT_BOUNDARY", prompt)
        self.assertNotIn("EXCLUDED_ABOVE_BOUNDARY", prompt)
        self.assertEqual(response["sources"], ["[notes.txt:paragraph-1]"])
        self.assertEqual(response["chunks_retrieved"], 1)
        self.assertEqual(collection.last_query["n_results"], 3)

    def test_no_match_skips_generation(self):
        with patch.object(rag, "generate_answer") as generate:
            response = self.ask(FixtureCollection([chunk(1.0), chunk(1.1, paragraph=2)]))
        generate.assert_not_called()
        self.assertEqual(response, {
            "answer": rag.NO_RELEVANT_ANSWER,
            "sources": [], "confidence": "low", "chunks_retrieved": 0,
        })

    def test_empty_collection_skips_generation(self):
        collection = FixtureCollection([])
        with patch.object(rag, "generate_answer") as generate:
            response = self.ask(collection)
        generate.assert_not_called()
        self.assertIsNone(collection.last_query)
        self.assertEqual(response["chunks_retrieved"], 0)
        self.assertEqual(response["confidence"], "low")

    def test_confidence_boundaries_use_best_distance(self):
        for distance, expected in [
            (0.4999, "high"), (0.5, "medium"), (0.9999, "medium"),
            (1.0, "low"), (1.2, "low"),
        ]:
            with self.subTest(distance=distance):
                self.assertEqual(rag.confidence_level([
                    chunk(distance + 0.1), chunk(distance, paragraph=2),
                ]), expected)
        self.assertEqual(rag.confidence_level([]), "low")

    def test_configurable_threshold_allows_low_match(self):
        with patch.object(rag, "generate_answer", return_value=("A fixture answer.", False)):
            response = self.ask(FixtureCollection([chunk(1.1)]), distance_threshold=1.5)
        self.assertEqual(response["confidence"], "low")
        self.assertEqual(response["chunks_retrieved"], 1)

    def test_stricter_threshold_can_reject_a_close_candidate(self):
        with patch.object(rag, "generate_answer") as generate:
            response = self.ask(FixtureCollection([chunk(0.4)]), distance_threshold=0.3)
        generate.assert_not_called()
        self.assertEqual(response["confidence"], "low")
        self.assertEqual(response["sources"], [])

    def test_invalid_thresholds_are_rejected(self):
        for threshold in [0, -1, float("nan"), float("inf"), "invalid"]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    rag.validate_threshold(threshold)

    def test_nonfinite_distances_never_reach_generation(self):
        with patch.object(rag, "generate_answer") as generate:
            response = self.ask(FixtureCollection([
                chunk(float("nan")), chunk(float("inf"), paragraph=2),
            ]))
        generate.assert_not_called()
        self.assertEqual(response["chunks_retrieved"], 0)

    def test_blank_query_returns_structured_error(self):
        with patch.object(rag, "generate_answer") as generate:
            response = self.ask(FixtureCollection([chunk(0.2)]), question=" ")
        generate.assert_not_called()
        self.assertIn("Please enter a question", response["answer"])
        self.assertEqual(response["confidence"], "low")

    def test_retrieval_failure_returns_structured_error(self):
        collection = FixtureCollection([chunk(0.2)])
        with patch.object(collection, "query", side_effect=RuntimeError("database unavailable")):
            response = self.ask(collection)
        self.assertIn("database unavailable", response["answer"])
        self.assertEqual(response["sources"], [])
        self.assertEqual(response["chunks_retrieved"], 0)

    def test_connection_failure_keeps_four_field_response(self):
        import requests
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
            response = self.ask(FixtureCollection([chunk(0.2)]))
        self.assertIn("Cannot connect to Ollama", response["answer"])
        self.assertEqual(set(response), {"answer", "sources", "confidence", "chunks_retrieved"})
        self.assertEqual(response["confidence"], "high")

    def test_all_four_query_types_print_structured_responses(self):
        # This checks reporting, not whether a real LLM follows instructions.
        collection = FixtureCollection([chunk(0.3)])
        transcript = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(rag, "BASE_DIR", Path(directory)), patch.object(
                rag, "generate_answer", return_value=("A fixture answer. [notes.txt:paragraph-1]", False),
            ) as generate, redirect_stdout(transcript):
                success = rag.run_tests(collection, rag.DEFAULT_MODEL, rag.DEFAULT_OLLAMA_URL)
            report = (Path(directory) / "test_results.md").read_text(encoding="utf-8")
        self.assertTrue(success)
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(transcript.getvalue().count("Structured response:"), 4)
        for category, question, _ in rag.TEST_QUESTIONS:
            self.assertIn(category, report)
            self.assertIn(question, report)

    def test_prompt_has_context_citations_and_uncertainty_rules(self):
        prompt = rag.build_rag_prompt("Explain it.", [chunk(0.2, text="A supported note.")])
        for part in ("<context>", "</context>", "[notes.txt:paragraph-1]", "A supported note.", "Explain it."):
            self.assertIn(part, prompt)
        for instruction in ("Never make up information", "I don't know", "Always cite sources"):
            self.assertIn(instruction, rag.SYSTEM_PROMPT)

    def test_clarity_check_has_no_retrieved_text(self):
        clarification = "Which application are you referring to?"
        with patch.object(rag, "_ollama_chat", return_value=(
            '{"needs_clarification": true, "clarifying_question": "'+clarification+'"}', False,
        )) as chat:
            answer, is_error = rag.generate_answer("How do I make it remember things?", [
                chunk(0.8, text="RETRIEVED_TEXT_MUST_NOT_ANCHOR_CLARITY"),
            ])
        self.assertFalse(is_error)
        self.assertEqual(answer, clarification)
        self.assertEqual(chat.call_count, 1)
        self.assertNotIn("RETRIEVED_TEXT_MUST_NOT_ANCHOR_CLARITY", str(chat.call_args))
        self.assertEqual(chat.call_args.kwargs["response_format"], rag.CLARITY_SCHEMA)

    def test_clear_query_continues_to_grounded_generation(self):
        with patch.object(rag, "_ollama_chat", side_effect=[
            ('{"needs_clarification": false, "clarifying_question": ""}', False),
            ("A supported answer. [notes.txt:paragraph-1]", False),
        ]) as chat:
            answer, is_error = rag.generate_answer("What is Python?", [chunk(0.2, text="GROUNDING_TEXT")])
        self.assertFalse(is_error)
        self.assertIn("[notes.txt:paragraph-1]", answer)
        self.assertEqual(chat.call_count, 2)
        self.assertIn("GROUNDING_TEXT", str(chat.call_args_list[1]))

    def test_invalid_clarity_check_stops_generation(self):
        for invalid in ('not JSON', '{}', '{"needs_clarification":"true", "clarifying_question":"Which tool?"}'):
            with self.subTest(invalid=invalid), patch.object(rag, "_ollama_chat", return_value=(invalid, False)) as chat:
                _, is_error = rag.generate_answer("What is Python?", [chunk(0.2)])
            self.assertTrue(is_error)
            self.assertEqual(chat.call_count, 1)

    def test_explicit_corpus_subject_skips_clarity_check(self):
        chunks = [chunk(0.2, source="frameworkalpha_notes.txt", text="FrameworkAlpha is a framework.")]
        with patch.object(rag, "_ollama_chat", return_value=("A grounded answer.", False)) as chat:
            _, is_error = rag.generate_answer("What is FrameworkAlpha, and how do I use it?", chunks)
        self.assertFalse(is_error)
        self.assertEqual(chat.call_count, 1)
        self.assertNotIn("response_format", chat.call_args.kwargs)
        self.assertIn("FrameworkAlpha is a framework.", str(chat.call_args))

    def test_later_topic_does_not_resolve_earlier_pronoun(self):
        chunks = [chunk(0.2, source="frameworkalpha_notes.txt", text="FrameworkAlpha is a framework.")]
        self.assertFalse(rag._question_names_subject("How do I use it in FrameworkAlpha?", chunks))

    def test_multiple_clarification_questions_use_one_fallback(self):
        with patch.object(rag, "_ollama_chat", return_value=(
            '{"needs_clarification":true,"clarifying_question":"Which tool? What should it do?"}', False,
        )):
            answer, is_error = rag.generate_answer("How do I use it?", [chunk(0.8)])
        self.assertFalse(is_error)
        self.assertEqual(answer, rag.DEFAULT_CLARIFICATION)
        self.assertEqual(answer.count("?"), 1)


if __name__ == "__main__":
    unittest.main()
