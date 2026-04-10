import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


def build_relation_definition(relation_name, relation_data):
    """
    Build an enriched textual definition for a relation.
    """
    definition = relation_data.get("def", "")

    type_pairs = relation_data.get("triplets", [])
    type_pairs_text = ", ".join(
        [f"({s}, {o})" for s, _, o in type_pairs[:30]]
    )

    enriched_definition = (
        f"Relation name: {relation_name}. "
        f"Definition: {definition}. "
        f"Possible type pairs: {type_pairs_text}."
    )

    return enriched_definition


def retrieve_relation_candidates(
    kg_path,
    relation_definitions_path,
    output_path=None,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    alpha=0.25,
    beta=0.25,
    gamma=0.50,
    top_k=5
):
    """
    Retrieve candidate schema relations for each triplet in a KG.

    Parameters
    ----------
    kg_path : str
        Path to KG JSON file containing extracted triplets.
    relation_definitions_path : str
        Path to relation definitions JSON file.
    output_path : str, optional
        Path to save enriched KG.
    model_name : str
        SentenceTransformer model name.
    alpha : float
        Weight for predicate vs relation name similarity.
    beta : float
        Weight for predicate vs enriched definition similarity.
    gamma : float
        Weight for triplet vs enriched definition similarity.
    top_k : int
        Number of candidate relations to keep.

    Returns
    -------
    list
        KG triplets enriched with candidate relations.
    """

    # Load data
    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    with open(relation_definitions_path, "r", encoding="utf-8") as f:
        relation_defs = json.load(f)

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Prepare relation representations
    relation_names = []
    relation_name_texts = []
    relation_definition_texts = []

    for relation_name, relation_data in relation_defs.items():
        enriched_definition = build_relation_definition(
            relation_name,
            relation_data
        )

        relation_names.append(relation_name)
        relation_name_texts.append(relation_name.replace("_", " "))
        relation_definition_texts.append(enriched_definition)

    # Precompute relation embeddings
    relation_name_embeddings = model.encode(
        relation_name_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    relation_definition_embeddings = model.encode(
        relation_definition_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    enriched_kg = []

    for triplet in tqdm(kg):

        subject = triplet.get("subject", "")
        predicate = triplet.get("predicate", "")
        obj = triplet.get("object", "")

        triplet_text = f"{subject} {predicate} {obj}"

        # Encode query texts
        predicate_embedding = model.encode(
            predicate,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        triplet_embedding = model.encode(
            triplet_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        candidates = []

        for idx, relation_name in enumerate(relation_names):

            # Score 1: predicate vs relation name
            pred_name_score = cosine_similarity(
                [predicate_embedding],
                [relation_name_embeddings[idx]]
            )[0][0]

            # Score 2: predicate vs enriched definition
            pred_def_score = cosine_similarity(
                [predicate_embedding],
                [relation_definition_embeddings[idx]]
            )[0][0]

            # Score 3: triplet vs enriched definition
            triplet_def_score = cosine_similarity(
                [triplet_embedding],
                [relation_definition_embeddings[idx]]
            )[0][0]

            final_score = (
                alpha * pred_name_score
                + beta * pred_def_score
                + gamma * triplet_def_score
            )

            candidates.append({
                "relation": relation_name,
                "score": round(float(final_score), 4),
                "pred_name_score": round(float(pred_name_score), 4),
                "pred_def_score": round(float(pred_def_score), 4),
                "triplet_def_score": round(float(triplet_def_score), 4)
            })

        # Sort candidates by score descending
        candidates = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        triplet["candidates_rel"] = candidates

        enriched_kg.append(triplet)

    # Save if requested
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_kg, f, indent=2, ensure_ascii=False)

    return enriched_kg

if __name__ == "__main__":
    kg_path = "../data/mm_kg.json"
    relation_definitions_path = "../data/semantic_relations_with_triplets_and_definitions.json"
    output_path = "../data/kg_with_relation_candidates.json"

    enriched_kg = retrieve_relation_candidates(
        kg_path=kg_path,
        relation_definitions_path=relation_definitions_path,
        output_path=output_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.25,
        beta=0.25,
        gamma=0.50,
        top_k=5
    )

    print(f"Number of triplets processed: {len(enriched_kg)}")
    print(f"Saved to: {output_path}")

    # Display first triplet and its candidates
    if len(enriched_kg) > 0:
        print(json.dumps(enriched_kg[0], indent=2, ensure_ascii=False))