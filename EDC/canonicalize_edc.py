"""
EDC-style Schema Canonicalization — avec sauvegarde incrémentale
Follows the Extract-Define-Canonicalize paper (Zhang & Soh, EMNLP 2024)

Chaque réponse LLM (Phase 2 et Phase 3) est sauvegardée immédiatement dans
des fichiers JSONL séparés. En cas de crash, le script reprend là où il
s'est arrêté sans rappeler l'API pour les triplets déjà traités.

Fichiers produits
─────────────────
  <output>                   → résultats finaux JSON
  <output>.phase2.jsonl      → toutes les réponses brutes Phase 2
  <output>.phase3.jsonl      → toutes les réponses brutes Phase 3
  <output>.checkpoint.jsonl  → résultats finalisés (reprise possible)

Usage:
  python canonicalize_edc.py \\
      --ontology  ontology.json \\
      --kg        kg_triplets.json \\
      --output    canonicalized.json \\
      --api_key   sk-... \\
      --top_k     5
"""

import json
import argparse
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

for _lib in ("httpx", "httpcore", "openai", "sentence_transformers", "transformers"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# ─── Utilitaires I/O ──────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict):
    """Ajoute un enregistrement en fin de fichier JSONL (une ligne = un call)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list:
    """Charge toutes les lignes valides d'un fichier JSONL existant."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Ligne JSONL corrompue ignorée dans %s", path)
    return records


def triplet_id(triplet: dict) -> str:
    """Identifiant stable d'un triplet (hash SHA-1 sur ses champs clés)."""
    key = json.dumps(
        {k: triplet.get(k, "") for k in ("sentence", "subject", "predicate", "object")},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]

# ─── DeepSeek client ──────────────────────────────────────────────────────────

def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def llm_call(client: OpenAI,
             prompt: str,
             max_tokens: int = 256,
             retries: int = 4) -> tuple:
    """
    Appelle l'API DeepSeek avec back-off exponentiel.

    Retourne
    --------
    (response_text, raw_meta)
      raw_meta contient le prompt complet, le modèle, les tokens utilisés,
      l'horodatage et l'éventuelle erreur — pour traçabilité totale.
    """
    meta = {
        "prompt":     prompt,
        "model":      "deepseek-chat",
        "max_tokens": max_tokens,
        "called_at":  now_iso(),
        "response":   None,
        "usage":      None,
        "error":      None,
        "attempts":   0,
    }

    for attempt in range(retries):
        meta["attempts"] = attempt + 1
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()
            meta["response"] = text
            meta["usage"] = {
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens":      resp.usage.total_tokens,
            }
            return text, meta

        except Exception as exc:
            wait = 2 ** attempt
            log.warning("API error (attempt %d/%d, attente %ds): %s",
                        attempt + 1, retries, wait, exc)
            meta["error"] = str(exc)
            time.sleep(wait)

    # Tous les essais épuisés → réponse vide mais meta sauvegardée
    meta["response"] = ""
    return "", meta

# ─── Phase 2 — Schema Definition ──────────────────────────────────────────────

DEFINITION_PROMPT = """\
Given a piece of text and a relational triplet extracted from it, write a \
concise one-sentence definition for the relation (predicate) used in the triplet.

The definition must describe the *semantic role* of the relation, not just \
paraphrase the sentence.

Text: {sentence}
Triplet: ['{subject}', '{predicate}', '{object}']

Write only the definition, nothing else."""


def generate_predicate_definition(client: OpenAI,
                                  triplet_uid: str,
                                  sentence: str,
                                  subject: str,
                                  predicate: str,
                                  obj: str,
                                  phase2_log: Path) -> str:
    """
    Phase 2 : génère une définition pour le prédicat extrait.
    Sauvegarde immédiatement le prompt + la réponse brute dans phase2_log.
    """
    prompt = DEFINITION_PROMPT.format(
        sentence=sentence,
        subject=subject,
        predicate=predicate,
        object=obj,
    )
    definition, meta = llm_call(client, prompt, max_tokens=150)

    # ── Sauvegarde immédiate ──────────────────────────────────────────────
    record = {
        "triplet_id": triplet_uid,
        "phase":      2,
        "predicate":  predicate,
        "sentence":   sentence,
        "subject":    subject,
        "object":     obj,
        **meta,               # prompt, model, response, usage, error, called_at
    }
    append_jsonl(phase2_log, record)
    log.debug("  [P2] '%s' → %s", predicate, definition[:80])
    return definition

# ─── Phase 3 — Schema Canonicalization ────────────────────────────────────────

CANONICALIZATION_PROMPT = """\
Given a piece of text, a relational triplet extracted from it, and the \
definition of its predicate, choose the most appropriate canonical relation \
from the choices below to replace the predicate in this context.

Text: {sentence}
Triplet: ['{subject}', '{predicate}', '{object}']
Definition of '{predicate}': {predicate_def}

Choices:
{choices}

Rules:
- Answer with ONLY the letter of the best match (e.g. A).
- Choose '{none_letter}' if no candidate is semantically equivalent in this \
context; do NOT over-generalise.
- Do not explain."""


def build_choices(candidates: list) -> tuple:
    """
    candidates : list of (relation_name, relation_definition)
    Retourne (texte des choix formaté, lettre pour 'None of the above')
    """
    lines = []
    for i, (rel, rel_def) in enumerate(candidates):
        letter = chr(65 + i)
        lines.append(f"{letter}. '{rel}': {rel_def}")
    none_letter = chr(65 + len(candidates))
    lines.append(f"{none_letter}. None of the above")
    return "\n".join(lines), none_letter


def canonicalize_with_llm(client: OpenAI,
                           triplet_uid: str,
                           sentence: str,
                           subject: str,
                           predicate: str,
                           obj: str,
                           predicate_def: str,
                           candidates: list,
                           phase3_log: Path):
    """
    Phase 3 : MCQ LLM pour choisir la relation canonique.
    Sauvegarde immédiatement le prompt + la réponse brute dans phase3_log.

    Retourne le nom de la relation choisie, ou None (= "None of the above").
    """
    choices_str, none_letter = build_choices(candidates)
    prompt = CANONICALIZATION_PROMPT.format(
        sentence=sentence,
        subject=subject,
        predicate=predicate,
        object=obj,
        predicate_def=predicate_def,
        choices=choices_str,
        none_letter=none_letter,
    )
    answer, meta = llm_call(client, prompt, max_tokens=10)
    answer_upper = answer.upper()

    # Interpréter la lettre choisie
    chosen_relation = None
    chosen_letter   = none_letter
    for i, (rel, _) in enumerate(candidates):
        letter = chr(65 + i)
        if letter in answer_upper:
            chosen_relation = rel
            chosen_letter   = letter
            break

    # ── Sauvegarde immédiate ──────────────────────────────────────────────
    record = {
        "triplet_id":      triplet_uid,
        "phase":           3,
        "predicate":       predicate,
        "sentence":        sentence,
        "subject":         subject,
        "object":          obj,
        "predicate_def":   predicate_def,
        "candidates":      [{"relation": r, "def": d} for r, d in candidates],
        "none_letter":     none_letter,
        "chosen_letter":   chosen_letter,
        "chosen_relation": chosen_relation,
        **meta,            # prompt, model, response, usage, error, called_at
    }
    append_jsonl(phase3_log, record)
    log.debug("  [P3] '%s' → '%s' (lettre %s)",
              predicate, chosen_relation or "None", chosen_letter)
    return chosen_relation

# ─── Embedding helpers ────────────────────────────────────────────────────────

def embed(model: SentenceTransformer, texts: list) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def top_k_by_similarity(query_vec: np.ndarray,
                         corpus_vecs: np.ndarray,
                         corpus_labels: list,
                         k: int) -> list:
    sims    = cosine_similarity(query_vec.reshape(1, -1), corpus_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [(corpus_labels[i], float(sims[i])) for i in top_idx]

# ─── Checkpoint ───────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Persiste chaque triplet finalisé dans un JSONL.
    Permet de reprendre le script sans repasser par l'API pour les triplets
    déjà traités.
    """

    def __init__(self, path: Path):
        self.path  = path
        self._done = {}
        self._load()

    def _load(self):
        for rec in load_jsonl(self.path):
            tid = rec.get("triplet_id")
            if tid:
                self._done[tid] = rec
        if self._done:
            log.info("Checkpoint : %d triplets rechargés depuis %s",
                     len(self._done), self.path)

    def is_done(self, tid: str) -> bool:
        return tid in self._done

    def get(self, tid: str) -> dict:
        return self._done[tid]

    def save(self, result: dict):
        tid = result["triplet_id"]
        self._done[tid] = result
        append_jsonl(self.path, result)

    def __len__(self):
        return len(self._done)

# ─── Cache de définitions ─────────────────────────────────────────────────────

class DefinitionCache:
    """
    Cache en mémoire des définitions générées (Phase 2).
    Pré-rempli depuis le log Phase 2 pour ne pas rappeler l'API.
    """

    def __init__(self):
        self._cache = {}

    def seed_from_phase2_log(self, path: Path):
        for rec in load_jsonl(path):
            pred = rec.get("predicate", "").lower().strip()
            defn = rec.get("response", "")
            if pred and defn and pred not in self._cache:
                self._cache[pred] = defn
        if self._cache:
            log.info("Cache Phase 2 : %d définitions rechargées", len(self._cache))

    def get(self, predicate: str):
        return self._cache.get(predicate.lower().strip())

    def set(self, predicate: str, definition: str):
        self._cache[predicate.lower().strip()] = definition

    def __len__(self):
        return len(self._cache)

# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_canonicalization(ontology: dict,
                         kg: list,
                         client: OpenAI,
                         output_path: Path,
                         top_k: int = 5,
                         embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                         ) -> list:
    """
    Phases 2 + 3 de l'article EDC avec sauvegarde incrémentale après chaque
    appel LLM et après chaque triplet finalisé.
    """

    # ── Chemins des fichiers annexes ──────────────────────────────────────
    phase2_log = output_path.with_suffix("").with_suffix("") \
                    if output_path.suffix == ".json" \
                    else output_path
    # Simplification : suffixes calculés proprement
    stem       = output_path.stem
    parent     = output_path.parent
    phase2_log = parent / f"{stem}.phase2.jsonl"
    phase3_log = parent / f"{stem}.phase3.jsonl"
    chk_path   = parent / f"{stem}.checkpoint.jsonl"

    log.info("Fichiers de log :")
    log.info("  Phase 2    → %s", phase2_log)
    log.info("  Phase 3    → %s", phase3_log)
    log.info("  Checkpoint → %s", chk_path)

    # ── Initialisation ────────────────────────────────────────────────────
    checkpoint = CheckpointManager(chk_path)
    def_cache  = DefinitionCache()
    def_cache.seed_from_phase2_log(phase2_log)

    # ── Ontologie ─────────────────────────────────────────────────────────
    onto_names = list(ontology.keys())
    onto_defs  = [ontology[r]["def"] for r in onto_names]
    log.info("Ontologie : %d relations", len(onto_names))

    # ── Embeddings de l'ontologie (calculés une seule fois) ───────────────
    log.info("Calcul des embeddings ontologie (modèle : %s)…", embed_model_name)
    embed_model = SentenceTransformer(embed_model_name)
    onto_vecs   = embed(embed_model, onto_defs)
    log.info("Embeddings shape : %s", onto_vecs.shape)

    # ── Boucle principale ─────────────────────────────────────────────────
    results      = []
    skipped      = 0
    phase2_calls = 0
    phase3_calls = 0

    for triplet in tqdm(kg, desc="Canonicalisation"):
        sentence  = triplet.get("sentence",  "")
        subject   = triplet.get("subject",   "")
        predicate = triplet.get("predicate", "")
        obj       = triplet.get("object",    "")
        tid       = triplet_id(triplet)

        # ── Triplet déjà traité lors d'une exécution précédente ──────────
        if checkpoint.is_done(tid):
            results.append(checkpoint.get(tid))
            skipped += 1
            continue

        # ── Phase 2 : définition du prédicat extrait ─────────────────────
        # Le cache évite de rappeler l'API pour un prédicat déjà vu
        pred_def = def_cache.get(predicate)
        if pred_def is None:
            pred_def = generate_predicate_definition(
                client, tid, sentence, subject, predicate, obj, phase2_log
            )
            def_cache.set(predicate, pred_def)
            phase2_calls += 1
        else:
            log.debug("  [P2] cache hit pour '%s'", predicate)

        # ── Phase 3a : top-k candidats par similarité vectorielle ─────────
        query_vec      = embed(embed_model, [pred_def])[0]
        top_candidates = top_k_by_similarity(
            query_vec, onto_vecs, onto_names, k=top_k
        )
        candidates_for_llm = [
            (rel, ontology[rel]["def"]) for rel, _score in top_candidates
        ]

        # ── Phase 3b : vérification LLM (MCQ) ───────────────────────────
        canonical = canonicalize_with_llm(
            client, tid,
            sentence, subject, predicate, obj,
            pred_def, candidates_for_llm, phase3_log,
        )
        phase3_calls += 1

        # ── Résultat finalisé ─────────────────────────────────────────────
        result = {
            "triplet_id":           tid,
            **triplet,
            "original_predicate":   predicate,
            "predicate_definition": pred_def,
            "canonical_predicate":  canonical if canonical else predicate,
            "canonicalized":        canonical is not None,
            "top_candidates": [
                {"relation": r, "similarity": round(s, 4)}
                for r, s in top_candidates
            ],
            "processed_at": now_iso(),
        }

        # ── Sauvegarde immédiate du triplet finalisé ──────────────────────
        checkpoint.save(result)
        results.append(result)

    # ── Rapport final ─────────────────────────────────────────────────────
    n_canon = sum(1 for r in results if r.get("canonicalized"))
    log.info("─" * 60)
    log.info("Triplets total          : %d", len(results))
    log.info("  dont repris (skip)    : %d", skipped)
    log.info("  dont nouveaux         : %d", len(results) - skipped)
    log.info("Canonicalisés           : %d / %d (%.1f%%)",
             n_canon, len(results), 100 * n_canon / max(len(results), 1))
    log.info("Appels LLM Phase 2      : %d  (cache hits: %d)",
             phase2_calls, len(results) - skipped - phase2_calls)
    log.info("Appels LLM Phase 3      : %d", phase3_calls)
    log.info("─" * 60)
    return results

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDC canonicalization (DeepSeek) avec sauvegarde incrémentale"
    )
    parser.add_argument("--ontology",    required=True,
                        help="JSON ontologie {relation: {def:...}}")
    parser.add_argument("--kg",          required=True,
                        help="JSON KG [{sentence, subject, predicate, object, ...}]")
    parser.add_argument("--output",      default="canonicalized.json",
                        help="Fichier JSON de sortie final (default: canonicalized.json)")
    parser.add_argument("--api_key",     required=True,
                        help="Clé API DeepSeek")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="Nombre de candidats pour le MCQ LLM (default: 5)")
    parser.add_argument("--embed_model",
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Modèle sentence-transformers (default: all-MiniLM-L6-v2)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Active les logs DEBUG")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Charger les entrées ────────────────────────────────────────────────
    log.info("Chargement de l'ontologie : %s", args.ontology)
    with open(args.ontology, encoding="utf-8") as f:
        ontology = json.load(f)

    log.info("Chargement du KG : %s", args.kg)
    with open(args.kg, encoding="utf-8") as f:
        kg_raw = json.load(f)

    # Normaliser le KG en liste de triplets
    if isinstance(kg_raw, list):
        kg = kg_raw
    elif isinstance(kg_raw, dict):
        kg = []
        for v in kg_raw.values():
            if isinstance(v, list):
                # Si les valeurs sont des listes de triplets
                for item in v:
                    if isinstance(item, dict):
                        kg.append(item)
            elif isinstance(v, dict) and "def" not in v:
                kg.append(v)
    log.info("KG normalisé : %d triplets", len(kg))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = make_client(args.api_key)

    # ── Lancer la canonicalisation ─────────────────────────────────────────
    results = run_canonicalization(
        ontology=ontology,
        kg=kg,
        client=client,
        output_path=output_path,
        top_k=args.top_k,
        embed_model_name=args.embed_model,
    )

    # ── Sauvegarder le JSON final complet ─────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Résultats finaux sauvegardés → %s", output_path)


if __name__ == "__main__":
    main()