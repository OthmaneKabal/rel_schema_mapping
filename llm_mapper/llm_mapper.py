import json
import ollama
from tqdm import tqdm


def build_candidate_block(candidates, relation_definitions):
    """
    Build a text block describing candidate relations and their definitions.
    """
    candidate_texts = []

    for idx, candidate in enumerate(candidates, start=1):
        rel_name = candidate["relation"]

        rel_def = relation_definitions.get(rel_name, {}).get("def", "")
        rel_triplets = relation_definitions.get(rel_name, {}).get("triplets", [])

        type_pairs = ", ".join(
            [f"({s}, {o})" for s, _, o in rel_triplets[:15]]
        )

        block = (
            f"{idx}. Relation: {rel_name}\n"
            f"Definition: {rel_def}\n"
            f"Possible type pairs: {type_pairs}\n"
            f"Retrieval score: {candidate['score']}\n"
        )

        candidate_texts.append(block)

    return "\n".join(candidate_texts)


def map_triplets_with_ollama(
    kg_candidates_path,
    relation_definitions_path,
    output_path=None,
    model_name="llama3"
):
    """
    Use an Ollama LLM to choose the best ontology relation
    among retrieved candidates for each triplet.

    Parameters
    ----------
    kg_candidates_path : str
        Path to KG JSON containing 'candidates_rel'.
    relation_definitions_path : str
        Path to ontology relation definitions JSON.
    output_path : str, optional
        Path to save mapped KG.
    model_name : str
        Ollama model name.

    Returns
    -------
    list
        KG with predicted relation added.
    """

    # Load files
    with open(kg_candidates_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    with open(relation_definitions_path, "r", encoding="utf-8") as f:
        relation_definitions = json.load(f)

    mapped_kg = []

    for idx, triplet in  enumerate(tqdm(kg, desc="Mapping triplets")):

        subject = triplet.get("subject", "")
        predicate = triplet.get("predicate", "")
        obj = triplet.get("object", "")
        sentence = triplet.get("sentence", "")
        candidates = triplet.get("candidates_rel", [])

        candidate_block = build_candidate_block(
            candidates,
            relation_definitions
        )

        prompt = f"""
You are an expert in ontology relation mapping.

Your goal is to map the extracted triplet to one ontology relation among the candidate relations.

Triplet:
- Subject: {subject}
- Predicate: {predicate}
- Object: {obj}

Source sentence:
{sentence}

Candidate ontology relations:
{candidate_block}

Instructions:
- Choose the single best ontology relation among the candidates.
- Use the relation definition and the source sentence to understand the meaning.
- If none of the candidate relations fit well, return: not_mapped
- Return only the relation name or not_mapped
- Do not explain your answer.

Answer:
"""

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            predicted_relation = response["message"]["content"].strip()

        except Exception as e:
            predicted_relation = "error"
            print(f"Error for triplet {idx}: {e}")

        triplet["predicted_relation"] = predicted_relation

        mapped_kg.append(triplet)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(kg)} triplets")

    # Save output if requested
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapped_kg, f, indent=2, ensure_ascii=False)

    return mapped_kg

if __name__ == "__main__":

    kg_candidates_path = "../data/kg_with_relation_candidates.json"
    relation_definitions_path = "../data/semantic_relations_with_triplets_and_definitions.json"
    output_path = "../data/kg_with_mapped_relations.json"

    mapped_kg = map_triplets_with_ollama(
        kg_candidates_path=kg_candidates_path,
        relation_definitions_path=relation_definitions_path,
        output_path=output_path,
        model_name="llama3.1:8b"
    )

    print(f"Number of triplets processed: {len(mapped_kg)}")
    print(f"Saved to: {output_path}")

    # Display first mapped triplet
    if len(mapped_kg) > 0:
        print(json.dumps(mapped_kg[0], indent=2, ensure_ascii=False))