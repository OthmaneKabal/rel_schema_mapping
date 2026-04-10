import pandas as pd
import json


def build_relation_triplets_with_definitions(srdef_path, semantic_network_path, output_json_path):
    """
    Build a JSON file mapping each semantic relation to:
      - all triplets in the semantic network using that relation
      - its definition from SRDEF

    Expected semantic network columns: subject, predicate, object
    Expected SRDEF relation rows: lines starting with 'RL|'

    Output structure:
    {
      "relation_name": {
        "triplets": [["s1", "relation_name", "o1"], ["s2", "relation_name", "o2"]],
        "def": "definition from SRDEF"
      },
      ...
    }
    """
    relation_defs = {}

    # Read SRDEF and extract relation definitions
    with open(srdef_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")

            if len(parts) >= 10 and parts[0] == "RL":
                rel_name = parts[2].strip()
                rel_def = parts[4].strip()
                inverse_name = parts[9].strip()

                if rel_name and rel_name not in relation_defs:
                    relation_defs[rel_name] = rel_def

                # store inverse relation too
                if inverse_name and inverse_name not in relation_defs:
                    relation_defs[inverse_name] = rel_def

    # Read semantic network
    sm = pd.read_csv(semantic_network_path)

    required_cols = {"subject", "predicate", "object"}
    missing = required_cols - set(sm.columns)
    if missing:
        raise ValueError(f"Missing required columns in semantic network file: {missing}")

    # Build result
    result = {}

    for relation, group in sm.groupby("predicate", dropna=True):
        triplets = group[["subject", "predicate", "object"]].astype(str).values.tolist()
        result[str(relation)] = {
            "triplets": triplets,
            "def": relation_defs.get(str(relation), "")
        }

    # Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

if __name__ == "__main__":
    result = build_relation_triplets_with_definitions(
        srdef_path="../data/SRDEF",
        semantic_network_path="../data/SM_network.csv",
        output_json_path="../data/semantic_relations_with_triplets_and_definitions.json"
    )

    print(f"Number of relations: {len(result)}")

    # Example: display first 5 relations
    for i, (rel, info) in enumerate(result.items()):
        print(f"\nRelation: {rel}")
        print(f"Definition: {info['def']}")
        print(f"Number of triplets: {len(info['triplets'])}")

        if len(info["triplets"]) > 0:
            print("Example triplet:", info["triplets"][0])

        if i >= 4:
            break