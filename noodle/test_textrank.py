import unittest

import spacy

from noodle.textrank import (Filter, GraphBuilder, KeyPhraseNormalizer,
                             TextRanker, Word)

nlp = spacy.load("en")


class TestTextRanker(unittest.TestCase):
    def test_should_parse_text(self):
        ranker = TextRanker(
            Filter(valid_pos=["VERB", "NOUN", "PROPN", "ADJ"]),
            GraphBuilder(window_size=2))

        text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
        phrases = ranker.parse(text, limit=10, threshold=0.1)

        self.assertGreater(len(phrases), 0)
        self.assertLessEqual(len(phrases), 10)
        self.assertEqual(phrases[0].text, "minimal generating sets")
        self.assertTrue(all(p.rank > 0.1 for p in phrases))

    def test_should_build_graph_from_list_of_words(self):
        builder = GraphBuilder(window_size=2)

        words = [
            Word(tok, i)
            for i, tok in enumerate(nlp("I am a python programmer"))
        ]
        graph = builder.build(words)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.number_of_nodes(), 5)
