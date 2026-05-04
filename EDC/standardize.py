import json
import pickle
import argparse
import re
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import nltk
nltk.data.path.insert(0, './nltk_data')

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_terms_from_excel(path, column="term", lowercase=False):
    df = pd.read_excel(path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    terms = []
    for value in df[column]:
        if pd.notna(value):
            term = str(value).strip()
            if term:
                if lowercase:
                    term = term.lower()
                terms.append(term)

    return terms


@lru_cache(maxsize=None)
def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s*'\s*", "'", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


@lru_cache(maxsize=None)
def node_cleanliness_score(node):
    if not node:
        return float("-inf")

    score = 0
    text = str(node).strip()

    if text == node:
        score += 2

    if any(c.isupper() for c in text):
        score += 3

    if text.islower():
        score -= 2

    words = text.split()
    if words:
        title_like = sum(1 for w in words if w and w[0].isupper())
        score += title_like * 0.5

    if re.search(r"\s+'\s*|\s*'\s+", text):
        score -= 3

    if re.search(r"\s-\s|\s-|-\s", text):
        score -= 3

    if re.search(r"\s{2,}", text):
        score -= 2

    punct_count = len(re.findall(r"[^\w\s']", text))
    score -= punct_count * 0.5

    score += min(len(text), 50) * 0.02

    return score


def choose_best_representative(variants, preferred_lookup=None):
    variants = list(set(v for v in variants if v is not None and str(v).strip()))

    if not variants:
        return None

    if preferred_lookup:
        for variant in variants:
            normalized = normalize_text(variant)
            if normalized in preferred_lookup:
                return preferred_lookup[normalized]

    return sorted(
        variants,
        key=lambda x: (node_cleanliness_score(x), len(x), x),
        reverse=True
    )[0]


def build_node_variant_groups(graph):
    groups = defaultdict(set)

    for triple in graph:
        for field in ["subject", "object"]:
            value = triple.get(field)

            if value is None:
                continue

            value = str(value).strip()
            if not value:
                continue

            normalized = normalize_text(value)
            if normalized:
                groups[normalized].add(value)

    return groups


def build_canonical_mapping(graph, preferred_terms=None):
    groups = build_node_variant_groups(graph)

    preferred_lookup = (
        {normalize_text(term): term for term in preferred_terms}
        if preferred_terms else None
    )

    replacement_map = {}

    for normalized, variants in groups.items():
        canonical = choose_best_representative(
            variants,
            preferred_lookup=preferred_lookup
        )

        for variant in variants:
            replacement_map[variant] = canonical

    return replacement_map


def canonicalize_graph_nodes(graph, preferred_terms=None):
    replacement_map = build_canonical_mapping(
        graph,
        preferred_terms=preferred_terms
    )

    corrected_graph = []

    for triple in tqdm(graph, desc="Canonicalizing node variants"):
        new_triple = triple.copy()

        subject = triple.get("subject")
        obj = triple.get("object")

        if subject in replacement_map:
            new_triple["subject"] = replacement_map[subject]

        if obj in replacement_map:
            new_triple["object"] = replacement_map[obj]

        corrected_graph.append(new_triple)

    return corrected_graph


def find_root(term, mapping):
    visited = set()
    current = term

    while current in mapping and mapping[current] != current and current not in visited:
        visited.add(current)
        current = mapping[current]

    return current


def correct_mapping_dict(mapping_dict, preferred_terms):
    corrected = mapping_dict.copy()

    preferred_terms = [
        str(t).strip()
        for t in preferred_terms
        if t is not None and str(t).strip()
    ]

    all_terms = set(corrected.keys()) | set(corrected.values())
    clusters = {}

    for term in all_terms:
        root = find_root(term, corrected)
        clusters.setdefault(root, set()).add(term)

    for root, cluster_terms in clusters.items():
        preferred_in_cluster = [p for p in preferred_terms if p in cluster_terms]

        if preferred_in_cluster:
            chosen = preferred_in_cluster[0]
            for term in cluster_terms:
                corrected[term] = chosen
            corrected[chosen] = chosen
        else:
            for term in cluster_terms:
                corrected[term] = root
            corrected[root] = root

    for preferred in preferred_terms:
        corrected[preferred] = preferred

    return corrected


def apply_mapping_and_remove_duplicates(graph, mapping):
    merged = {}

    for triple in tqdm(graph, desc="Applying mapping and removing duplicates"):
        subject = mapping.get(triple.get("subject", ""), triple.get("subject", ""))
        predicate = triple.get("predicate", "")
        obj = mapping.get(triple.get("object", ""), triple.get("object", ""))

        key = (subject, predicate, obj)

        if key not in merged:
            new_triple = triple.copy()
            new_triple["subject"] = subject
            new_triple["object"] = obj
            merged[key] = new_triple
        else:
            old_sentence = merged[key].get("sentence", "")
            new_sentence = triple.get("sentence", "")

            if new_sentence and new_sentence not in old_sentence:
                merged[key]["sentence"] = (
                    old_sentence + " | " + new_sentence
                    if old_sentence else new_sentence
                )

    return list(merged.values())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_kg", required=True)
    parser.add_argument("--mapping_pickle", required=True)
    parser.add_argument("--preferred_excel", required=True)
    parser.add_argument("--term_column", default="term")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    graph = read_json(args.input_kg)
    mapping_dict = read_pickle(args.mapping_pickle)
    preferred_terms = read_terms_from_excel(args.preferred_excel, args.term_column)

    print(f"Input triples: {len(graph)}")
    print(f"Preferred terms: {len(preferred_terms)}")

    graph = canonicalize_graph_nodes(graph, preferred_terms)

    corrected_mapping = correct_mapping_dict(
        mapping_dict=mapping_dict,
        preferred_terms=preferred_terms
    )

    final_graph = apply_mapping_and_remove_duplicates(
        graph=graph,
        mapping=corrected_mapping
    )

    save_json(final_graph, args.output)

    print(f"Final triples: {len(final_graph)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()