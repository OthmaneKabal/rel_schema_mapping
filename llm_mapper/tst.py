import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def build_candidate_block(candidates, relation_definitions):
    candidate_texts = []
    for idx, candidate in enumerate(candidates, start=1):
        rel_name     = candidate["relation"]
        rel_def      = relation_definitions.get(rel_name, {}).get("def", "")[:150]
        rel_triplets = relation_definitions.get(rel_name, {}).get("triplets", [])
        type_pairs   = ", ".join([f"({s}, {o})" for s, _, o in rel_triplets[:5]])
        block = (
            f"{idx}. Relation: {rel_name}\n"
            f"Definition: {rel_def}\n"
            f"Possible type pairs: {type_pairs}\n"
            f"Retrieval score: {candidate['score']}\n"
        )
        candidate_texts.append(block)
    return "\n".join(candidate_texts)


def build_prompt(subject, predicate, obj, sentence, candidate_block):
    return (
        "You are an expert in ontology relation mapping.\n\n"
        "Your goal is to map the extracted triplet to one ontology relation among the candidate relations.\n\n"
        "Triplet:\n"
        f"- Subject: {subject}\n"
        f"- Predicate: {predicate}\n"
        f"- Object: {obj}\n\n"
        f"Source sentence:\n{sentence}\n\n"
        f"Candidate ontology relations:\n{candidate_block}\n\n"
        "Instructions:\n"
        "- Choose the single best ontology relation among the candidates.\n"
        "- Use the relation definition and the source sentence to understand the meaning.\n"
        "- If none of the candidate relations fit well, return: not_mapped\n"
        "- Return only the relation name or not_mapped\n"
        "- Do not explain your answer.\n\n"
        "Answer:"
    )


def save_checkpoint(kg, predictions, output_path):
    partial_kg = []
    for triplet, pred in zip(kg[:len(predictions)], predictions):
        t = dict(triplet)
        t["predicted_relation"] = pred
        partial_kg.append(t)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(partial_kg, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Checkpoint saved ({len(predictions)}/{len(kg)}) → {output_path}")


# ──────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────

def map_triplets_with_vllm(
    kg_candidates_path: str,
    relation_definitions_path: str,
    output_path: str = None,
    model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size: int = 1,
    max_new_tokens: int = 16,
    batch_size: int = 100,       # checkpoint toutes les N inférences
):
    # ── Load data ──────────────────────────────
    print(f"Loading KG from: {kg_candidates_path}")
    with open(kg_candidates_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    print(f"Loading relation definitions from: {relation_definitions_path}")
    with open(relation_definitions_path, "r", encoding="utf-8") as f:
        relation_definitions = json.load(f)

    # ── Resume from checkpoint if exists ───────
    all_predictions = []
    start_index = 0

    if output_path and Path(output_path).exists():
        print(f"Found existing checkpoint at {output_path}, resuming...")
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        start_index = len(existing)
        all_predictions = [t["predicted_relation"] for t in existing]
        print(f"  → Resuming from triplet {start_index}/{len(kg)}")

    # ── Load tokenizer ─────────────────────────
    # Opérations string uniquement — pas de multiprocessing, pas de conflit avec vLLM
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # ── Build all prompts sequentially ─────────
    print(f"Building {len(kg) - start_index} prompts...")
    all_prompts = []
    for triplet in tqdm(kg[start_index:], desc="Building prompts"):
        candidate_block = build_candidate_block(
            triplet.get("candidates_rel", []), relation_definitions
        )
        raw_prompt = build_prompt(
            subject         = triplet.get("subject",   ""),
            predicate       = triplet.get("predicate", ""),
            obj             = triplet.get("object",    ""),
            sentence        = triplet.get("sentence",  ""),
            candidate_block = candidate_block,
        )
        if tokenizer.chat_template is not None:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = raw_prompt
        all_prompts.append(formatted)

    del tokenizer  # libère mémoire avant de charger vLLM

    # ── Load vLLM engine ───────────────────────
    print(f"Loading vLLM engine: {model_name_or_path}")
    llm = LLM(
        model                  = model_name_or_path,
        tensor_parallel_size   = tensor_parallel_size,
        dtype                  = "bfloat16",
        gpu_memory_utilization = 0.90,
        max_model_len          = 4096,
        enable_prefix_caching  = True,
    )

    sampling_params = SamplingParams(
        temperature = 0.0,
        max_tokens  = max_new_tokens,
        stop        = ["\n"],
    )

    # ── Inference par batch + checkpoint toutes les 100 ──
    print(f"Running inference ({len(all_prompts)} prompts, batch_size={batch_size})...")

    for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc="Batches"):
        batch = all_prompts[batch_start : batch_start + batch_size]

        outputs = llm.generate(batch, sampling_params)

        for o in outputs:
            text = o.outputs[0].text.strip().split("\n")[0].strip()
            all_predictions.append(text)

        # Sauvegarde après chaque batch
        if output_path:
            save_checkpoint(kg, all_predictions, output_path)

    # ── Merge final ────────────────────────────
    mapped_kg = []
    for triplet, pred in zip(kg, all_predictions):
        t = dict(triplet)
        t["predicted_relation"] = pred
        mapped_kg.append(t)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapped_kg, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Final save → {output_path}")

    return mapped_kg


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":

    kg_candidates_path        = "../data/kg_with_relation_candidates.json"
    relation_definitions_path = "../data/semantic_relations_with_triplets_and_definitions.json"
    output_path               = "../data/kg_with_mapped_relations.json"

    mapped_kg = map_triplets_with_vllm(
        kg_candidates_path        = kg_candidates_path,
        relation_definitions_path = relation_definitions_path,
        output_path               = output_path,
        model_name_or_path        = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size      = 1,
        max_new_tokens            = 16,
        batch_size                = 100,   # checkpoint toutes les 100
    )

    print(f"\nDone. {len(mapped_kg)} triplets processed.")
    if mapped_kg:
        print(json.dumps(mapped_kg[0], indent=2, ensure_ascii=False))
