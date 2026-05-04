import json
from tqdm import tqdm
import pandas as pd
import json
import networkx as nx
from collections import defaultdict, deque, Counter
import argparse
import json
import re
from pathlib import Path
import numpy as np

def read_json_file(file_path):
    """
    Read and return the content of a JSON file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    dict or list
        Parsed JSON content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

def csv_to_json_triplets(csv_path, output_json_path=None):
    """
    Convert a CSV with columns [subject, predicate, object]
    into a JSON list of triplets.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    output_json_path : str, optional
        Path to save the JSON file.

    Returns
    -------
    list
        List of dictionaries with keys:
        subject, predicate, object
    """
    df = pd.read_csv(csv_path)

    required_cols = ["subject", "predicate", "object"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    triplets = df[required_cols].to_dict(orient="records")

    if output_json_path is not None:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(triplets, f, indent=2, ensure_ascii=False)

    return triplets
def replace_predicate_with_canonical(input_path, output_path):
    """
    Read a graph JSON file, replace 'predicate' by 'canonical_predicate',
    and save the updated graph.
    
    Parameters
    ----------
    input_path : str
        Path to the input JSON file.
    output_path : str
        Path to the output JSON file.
    
    Returns
    -------
    list[dict]
        Updated graph.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    updated_graph = []

    for triple in graph:
        triple = triple.copy()

        canonical_pred = triple.get("canonical_predicate")

        if canonical_pred and str(canonical_pred).strip():
            triple["original_predicate"] = triple.get("predicate")
            triple["predicate"] = canonical_pred

        updated_graph.append(triple)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_graph, f, indent=2, ensure_ascii=False)

    return updated_graph

import json

def restore_if_low_similarity(data, output_path, threshold=0.4):
    cp = 0

    for triplet in data:
        if triplet.get("canonicalized", False):

            canonical = triplet.get("canonical_predicate")
            original = triplet.get("original_predicate")

            similarity = None
            for c in triplet.get("top_candidates", []):
                if c["relation"] == canonical:
                    similarity = c["similarity"]
                    break

            if similarity is None or similarity < threshold:
                triplet["predicate"] = original
                triplet["canonicalized"] = False
            else:
                triplet["predicate"] = canonical
                cp += 1

            triplet["canonical_similarity"] = similarity

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved KG → {output_path}")
    print(f"Kept canonical predicates: {cp}")

    return data
# {
#     "triplet_id": "e1db2bc9c7c44a18",
#     "sentence": "Precision was determined at five different concentrations ranging from 0.25 to 1.5 ng/mL with 15 replicates at each level for whole blood and demonstrated a < 2.4 % coefficient of variation ( CV ) .",
#     "subject": "precision",
#     "predicate": "evaluation_of",
#     "object": "five different concentration",
#     "validation": "True",
#     "original_predicate": "be-solve-at",
#     "predicate_definition": "The relation indicates the specific experimental conditions or levels at which a measurement or property (precision) was evaluated.",
#     "canonical_predicate": "evaluation_of",
#     "canonicalized": true,
#     "top_candidates": [
    #   {
    #     "relation": "degree_of",
    #     "similarity": 0.5441
    #   },
    #   {
    #     "relation": "measures",
    #     "similarity": 0.4998
    #   },
    #   {
    #     "relation": "measurement_of",
    #     "similarity": 0.4916
    #   },
    #   {
    #     "relation": "evaluation_of",
    #     "similarity": 0.4046
    #   },
    #   {
    #     "relation": "assesses_effect_of",
    #     "similarity": 0.3938
    #   }
# ]

def merge_self_mapped_kg(
    target_aligned_path: str,
    self_canonicalized_path: str,
    output_path: str,
) -> list:
    """
    Fusionne le KG target-aligned et le KG self-canonicalisé en un KG unique.

    Les triplets non mappés (canonicalized=False) sont remplacés par leur
    version self-canonicalisée. Les triplets déjà mappés vers l'ontologie
    sont conservés tels quels.

    Args:
        target_aligned_path    : output de canonicalize_edc_mt.py
        self_canonicalized_path: output de self_canonicalize_edc.py
        output_path            : chemin du KG final fusionné

    Returns:
        Liste des triplets du KG final.
    """
    with open(target_aligned_path, encoding="utf-8") as f:
        target_aligned = json.load(f)

    with open(self_canonicalized_path, encoding="utf-8") as f:
        self_canonicalized = json.load(f)

    # Index des self-canonicalisés par triplet_id
    self_map = {r["triplet_id"]: r for r in self_canonicalized}

    # Fusion
    final_kg = []
    for triplet in target_aligned:
        tid = triplet["triplet_id"]
        final_kg.append(self_map.get(tid, triplet))

    # Stats
    n_onto = sum(1 for t in final_kg if t["triplet_id"] not in self_map)
    n_self = sum(1 for t in final_kg if t["triplet_id"] in self_map)
    n_fused = sum(1 for t in final_kg if t["triplet_id"] in self_map and t.get("canonicalized"))
    n_added = sum(1 for t in final_kg if t["triplet_id"] in self_map and not t.get("canonicalized"))

    print(f"KG final : {len(final_kg)} triplets")
    print(f"  mappés vers ontologie      : {n_onto}")
    print(f"  self-canonicalisés         : {n_self}")
    print(f"    dont fusionnés           : {n_fused}")
    print(f"    dont ajoutés au schéma   : {n_added}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_kg, f, indent=2, ensure_ascii=False)

    print(f"Sauvegardé → {output_path}")
    return final_kg
    
def nb_unique_predicates(data):
    predicates = set()
    for e in data:
        predicates.add(e["predicate"])
    return len(predicates)

import json

def merge_and_normalize_kg(file1, file2, output_file):
    """
    Merge two KG JSON files, normalize 'isa' -> 'is-a', and save result.

    Parameters
    ----------
    file1 : str
        Path to first KG JSON file
    file2 : str
        Path to second KG JSON file
    output_file : str
        Path to save merged KG
    """

    # Load files
    with open(file1, 'r') as f:
        kg1 = json.load(f)

    with open(file2, 'r') as f:
        kg2 = json.load(f)

    # Merge
    merged_kg = kg1 + kg2

    # Normalize predicates
    count = 0
    for triple in merged_kg:
        pred = triple.get("predicate", "").lower().strip()

        # Normalize isa / is → is-a
        if pred in {"is-a", "is"}:
            triple["predicate"] = "isa"
            count += 1

    # Save
    with open(output_file, 'w') as f:
        json.dump(merged_kg, f, indent=2)

    print(f"Saved merged KG to {output_file}")
    print(f"{count} predicates normalized (isa/is → is-a)")

    return merged_kg


if __name__ =="__main__":

    data = read_json_file("../data/graph_before_mapping/noisy_kg_before_mapping_Entity_Mapped_corrected_vf.json")
    terms  = set([i.strip().lower() for i in list(pd.read_excel("../data/common_nodes.xlsx").term)])
    print(len(terms))
    entities = set()
    for element in data:
        entities.add(element["subject"].strip().lower())            # main()
        entities.add(element["object"].strip().lower())
    print(len(entities))
    print(len(terms.intersection(entities)))
    




    # merge_and_normalize_kg("EDC_canonicalized_kg_v2.json","../data/is_a_augmentation_MM_mapped_nci_All_R_KG.json","EDC_canonicalized_kg_v2_augmented.json")
    #replace_predicate_with_canonical("EDC_canonicalized_v2.json","EDC_canonicalized_kg.json")
    # merge_self_mapped_kg(
    #     target_aligned_path="canonicalized_vf.json",
    #     self_canonicalized_path="self_canonicalized_.json",
    #     output_path="EDC_canonicalized_v2.json",
    # )


    ####################### Get unmapped ###############
    # # new_triplets = restore_if_low_similarity(data,"canonicalized_0.4_vf_kg.json",0.4)
    # with open('canonicalized_vf_kg.json') as f:
    #     data = json.load(f)

    # # Filtrer les triplets non mappés
    # non_mapped = [
    #     r for r in data
    #     if not r['canonicalized']
    #     and not r['predicate_definition'].lower().startswith('(already')
    # ]

    # # Sauvegarder le résultat
    # with open('non_mapped_.json', 'w') as f:
    #     json.dump(non_mapped, f, indent=2)

    # # Afficher le nombre de triplets non mappés
    # print(len(non_mapped), 'triplets non mappés')