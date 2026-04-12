import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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


def load_model(model_name_or_path: str, device: str = "auto"):
    """
    Load tokenizer and model from HuggingFace or a local path.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local directory path.
        e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct" or "./models/llama3"
    device : str
        "auto", "cpu", "cuda", or "mps"

    Returns
    -------
    tokenizer, model
    """
    print(f"Loading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading model from: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully.")
    return tokenizer, model


def generate_response(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    """
    Generate a response from the model given a prompt.
    """
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def map_triplets_with_transformers(
    kg_candidates_path: str,
    relation_definitions_path: str,
    output_path: str = None,
    model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device: str = "auto",
    max_new_tokens: int = 64,
    checkpoint_every: int = 500,
):
    """
    Use a local HuggingFace LLM to choose the best ontology relation
    among retrieved candidates for each triplet.

    Parameters
    ----------
    kg_candidates_path : str
        Path to KG JSON containing 'candidates_rel'.
    relation_definitions_path : str
        Path to ontology relation definitions JSON.
    output_path : str, optional
        Path to save the mapped KG.
    model_name_or_path : str
        HuggingFace model ID or local directory path.
    device : str
        Device to use: "auto", "cpu", "cuda", "mps".
    max_new_tokens : int
        Max tokens to generate per prediction.
    checkpoint_every : int
        Save a checkpoint to output_path every N triplets.

    Returns
    -------
    list
        KG with predicted relation added.
    """

    with open(kg_candidates_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    with open(relation_definitions_path, "r", encoding="utf-8") as f:
        relation_definitions = json.load(f)

    tokenizer, model = load_model(model_name_or_path, device=device)

    mapped_kg = []

    for idx, triplet in enumerate(tqdm(kg, desc="Mapping triplets")):

        subject    = triplet.get("subject", "")
        predicate  = triplet.get("predicate", "")
        obj        = triplet.get("object", "")
        sentence   = triplet.get("sentence", "")
        candidates = triplet.get("candidates_rel", [])

        candidate_block = build_candidate_block(candidates, relation_definitions)

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
            predicted_relation = generate_response(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )

            # Garder uniquement la première ligne si le modèle génère trop
            predicted_relation = predicted_relation.split("\n")[0].strip()

        except Exception as e:
            predicted_relation = "error"
            print(f"Error for triplet {idx}: {e}")

        triplet["predicted_relation"] = predicted_relation
        mapped_kg.append(triplet)

        # Sauvegarde intermédiaire toutes les `checkpoint_every` triplets
        if (idx + 1) % checkpoint_every == 0:
            print(f"Processed {idx + 1}/{len(kg)} triplets")
            if output_path is not None:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(mapped_kg, f, indent=2, ensure_ascii=False)
                print(f"  -> Checkpoint saved to: {output_path}")

    # Sauvegarde finale
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapped_kg, f, indent=2, ensure_ascii=False)
        print(f"Final save -> {output_path}")

    return mapped_kg


if __name__ == "__main__":

    kg_candidates_path        = "../data/kg_with_relation_candidates.json"
    relation_definitions_path = "../data/semantic_relations_with_triplets_and_definitions.json"
    output_path               = "../data/kg_with_mapped_relations.json"

    mapped_kg = map_triplets_with_transformers(
        kg_candidates_path=kg_candidates_path,
        relation_definitions_path=relation_definitions_path,
        output_path=output_path,
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        device="auto",
        max_new_tokens=64,
        checkpoint_every=10,   # <- modifiable ici
    )

    print(f"Number of triplets processed: {len(mapped_kg)}")
    print(f"Saved to: {output_path}")

    if len(mapped_kg) > 0:
        print(json.dumps(mapped_kg[0], indent=2, ensure_ascii=False))