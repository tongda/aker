"""
Text Ranker
"""

from collections import namedtuple
from typing import Dict, List

import networkx as nx
import spacy

nlp = spacy.load("en")

Word = namedtuple("Word", ["token", "index"])
RankedPhrase = namedtuple("RankedPhrase", ["text", "rank"])


class Filter(object):
    def __init__(self, valid_pos: List[str]):
        self.valid_pos = valid_pos

    def is_keep(self, word: Word):
        return not word.token.is_stop and word.token.pos_ in self.valid_pos


class GraphBuilder(object):
    def __init__(self, window_size: int):
        self.window_size = window_size

    def build(self, words: List[Word]) -> nx.DiGraph:
        graph = nx.DiGraph()

        for w1, w2 in self.iter_tiles(words):
            lem1 = w1.token.lemma_
            lem2 = w2.token.lemma_
            if not graph.has_node(lem1):
                graph.add_node(lem1)

            if not graph.has_node(lem2):
                graph.add_node(lem2)

            if graph.has_edge(lem1, lem2):
                graph.edges[lem1, lem2]['weights'] += 1.
            else:
                graph.add_edge(lem1, lem2, weight=1.0)

        return graph

    def iter_tiles(self, words: List[Word]):
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words[i + 1:i + 1 + self.window_size]):
                if w2.index - w1.index < self.window_size:
                    yield w1, w2


class KeyPhraseNormalizer(object):
    def __init__(self, ranks: Dict[str, float]):
        self.ranks = ranks

    def normalize(self, doc: spacy.tokens.doc.Doc):
        # last_idx = 0
        # for i, tok in enumerate(doc):
        #     if tok.lemma_ not in self.ranks:
        #         if i > last_idx:
        #             subtext = " ".join(t.text for t in doc[last_idx: i])
        #             print("subtext: ", subtext)
        #             yield from nlp(subtext).noun_chunks
        #             last_idx = i + 1
        #         else:
        #             last_idx += 1
        # if last_idx < len(doc):
        #     subtext = " ".join(t.text for t in doc[last_idx:])
        #     print("subtext: ", subtext)
        #     yield from nlp(subtext).noun_chunks
        for chunk in doc.noun_chunks:
            if chunk[0].pos_ == "DET":
                chunk = chunk[1:]
            if all(tok.lemma_ in self.ranks for tok in chunk):
                yield chunk


class TextRanker(object):
    def __init__(self, filter: Filter, graph_builder: GraphBuilder):
        self.filter = filter
        self.graph_builder = graph_builder
        self.normalizer = None
        self.ranks = None
        self.graph = None

    def parse(self, text: str, limit=10, threshold=0.1):
        doc = nlp(text)
        words = [Word(tok, idx) for idx, tok in enumerate(doc)]
        valid_words = [w for w in words if self.filter.is_keep(w)]
        self.graph = self.graph_builder.build(valid_words)
        self.ranks = nx.pagerank(self.graph)
        self.normalizer = KeyPhraseNormalizer(self.ranks)

        phrases = []
        for phrase in self.normalizer.normalize(doc):
            text = " ".join(tok.text for tok in phrase)
            rank = sum(self.ranks[tok.lemma_] for tok in phrase)
            if rank > threshold:
                phrases.append(RankedPhrase(text, rank))

        return sorted(phrases, key=lambda p: p.rank, reverse=True)[:limit]
