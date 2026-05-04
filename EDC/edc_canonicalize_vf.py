# """
# EDC-style Schema Canonicalization — sans cache de définitions
# Follows the Extract-Define-Canonicalize paper (Zhang & Soh, EMNLP 2024)

# Différences par rapport à la version avec cache :
#   - Phase 2 : le LLM est appelé pour CHAQUE triplet individuellement,
#     même si le prédicat a déjà été vu → définition contextuelle correcte
#   - Court-circuit : les prédicats déjà présents dans l'ontologie ne
#     passent pas par le LLM (fidèle au code officiel)
#   - Parsing MCQ : regex stricte sur la première lettre isolée pour éviter
#     les faux positifs ("Neither A nor B" → ne prend plus A)

# Fichiers produits
# ─────────────────
#   <output>                   → résultats finaux JSON
#   <output>.phase2.jsonl      → toutes les réponses brutes Phase 2
#   <output>.phase3.jsonl      → toutes les réponses brutes Phase 3
#   <output>.checkpoint.jsonl  → résultats finalisés (reprise possible)

# Usage:
#   python canonicalize_edc_no_cache.py \\
#       --ontology  ontology.json \\
#       --kg        kg_triplets.json \\
#       --output    canonicalized.json \\
#       --api_key   sk-... \\
#       --top_k     5
# """

# import json
# import re
# import argparse
# import logging
# import time
# import hashlib
# from pathlib import Path
# from datetime import datetime, timezone

# import numpy as np
# from openai import OpenAI
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm

# # ─── Logging ──────────────────────────────────────────────────────────────────

# logging.basicConfig(level=logging.WARNING)
# log = logging.getLogger(__name__)

# for _lib in ("httpx", "httpcore", "openai", "sentence_transformers", "transformers"):
#     logging.getLogger(_lib).setLevel(logging.ERROR)

# # ─── Utilitaires I/O ──────────────────────────────────────────────────────────

# def now_iso() -> str:
#     return datetime.now(timezone.utc).isoformat()


# def append_jsonl(path: Path, record: dict):
#     """Ajoute un enregistrement en fin de fichier JSONL."""
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


# def load_jsonl(path: Path) -> list:
#     """Charge toutes les lignes valides d'un fichier JSONL existant."""
#     if not path.exists():
#         return []
#     records = []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 try:
#                     records.append(json.loads(line))
#                 except json.JSONDecodeError:
#                     log.warning("Ligne JSONL corrompue ignorée dans %s", path)
#     return records


# def triplet_id(triplet: dict) -> str:
#     """Identifiant stable d'un triplet (hash SHA-1 sur ses champs clés)."""
#     key = json.dumps(
#         {k: triplet.get(k, "") for k in ("sentence", "subject", "predicate", "object")},
#         sort_keys=True, ensure_ascii=False,
#     )
#     return hashlib.sha1(key.encode()).hexdigest()[:16]

# # ─── DeepSeek client ──────────────────────────────────────────────────────────

# def make_client(api_key: str) -> OpenAI:
#     return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# def llm_call(client: OpenAI,
#              prompt: str,
#              max_tokens: int = 256,
#              retries: int = 4) -> tuple:
#     """
#     Appelle l'API DeepSeek avec back-off exponentiel.

#     Retourne (response_text, raw_meta).
#     """
#     meta = {
#         "prompt":     prompt,
#         "model":      "deepseek-chat",
#         "max_tokens": max_tokens,
#         "called_at":  now_iso(),
#         "response":   None,
#         "usage":      None,
#         "error":      None,
#         "attempts":   0,
#     }

#     for attempt in range(retries):
#         meta["attempts"] = attempt + 1
#         try:
#             resp = client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0,
#                 max_tokens=max_tokens,
#             )
#             text = resp.choices[0].message.content.strip()
#             meta["response"] = text
#             meta["usage"] = {
#                 "prompt_tokens":     resp.usage.prompt_tokens,
#                 "completion_tokens": resp.usage.completion_tokens,
#                 "total_tokens":      resp.usage.total_tokens,
#             }
#             return text, meta

#         except Exception as exc:
#             wait = 2 ** attempt
#             log.warning("API error (attempt %d/%d, attente %ds): %s",
#                         attempt + 1, retries, wait, exc)
#             meta["error"] = str(exc)
#             time.sleep(wait)

#     meta["response"] = ""
#     return "", meta

# # ─── Phase 2 — Schema Definition ──────────────────────────────────────────────

# DEFINITION_PROMPT = """\
# Given a piece of text and a relational triplet extracted from it, write a \
# concise one-sentence definition for the relation (predicate) used in the triplet.

# The definition must describe the *semantic role* of the relation in THIS \
# specific context, not just paraphrase the sentence.

# Text: {sentence}
# Triplet: ['{subject}', '{predicate}', '{object}']

# Write only the definition, nothing else."""


# def generate_predicate_definition(client: OpenAI,
#                                   triplet_uid: str,
#                                   sentence: str,
#                                   subject: str,
#                                   predicate: str,
#                                   obj: str,
#                                   phase2_log: Path) -> str:
#     """
#     Phase 2 : génère une définition CONTEXTUELLE pour le prédicat du triplet.
#     Appelé pour chaque triplet sans exception — pas de cache.
#     Sauvegarde immédiatement le prompt + la réponse brute dans phase2_log.
#     """
#     prompt = DEFINITION_PROMPT.format(
#         sentence=sentence,
#         subject=subject,
#         predicate=predicate,
#         object=obj,
#     )
#     definition, meta = llm_call(client, prompt, max_tokens=150)

#     record = {
#         "triplet_id": triplet_uid,
#         "phase":      2,
#         "predicate":  predicate,
#         "sentence":   sentence,
#         "subject":    subject,
#         "object":     obj,
#         **meta,
#     }
#     append_jsonl(phase2_log, record)
#     log.debug("  [P2] '%s' → %s", predicate, definition[:80])
#     return definition

# # ─── Phase 3 — Schema Canonicalization ────────────────────────────────────────

# CANONICALIZATION_PROMPT = """\
# Given a piece of text, a relational triplet extracted from it, and the \
# definition of its predicate, choose the most appropriate canonical relation \
# from the choices below to replace the predicate in this context.

# Text: {sentence}
# Triplet: ['{subject}', '{predicate}', '{object}']
# Definition of '{predicate}': {predicate_def}

# Choices:
# {choices}

# Rules:
# - Answer with ONLY the letter of the best match (e.g. A).
# - Choose '{none_letter}' if no candidate is semantically equivalent in this \
# context; do NOT over-generalise.
# - Do not explain."""


# def build_choices(candidates: list) -> tuple:
#     """
#     candidates : list of (relation_name, relation_definition)
#     Retourne (texte des choix formaté, lettre pour 'None of the above')
#     """
#     lines = []
#     for i, (rel, rel_def) in enumerate(candidates):
#         letter = chr(65 + i)
#         lines.append(f"{letter}. '{rel}': {rel_def}")
#     none_letter = chr(65 + len(candidates))
#     lines.append(f"{none_letter}. None of the above")
#     return "\n".join(lines), none_letter


# def parse_mcq_answer(answer: str, none_letter: str, n_candidates: int) -> str | None:
#     """
#     Parse la réponse MCQ du LLM de façon robuste.

#     Cherche une lettre isolée (A, B, C...) en début de réponse ou
#     après des marqueurs courants ("Answer:", "The answer is", etc.).
#     Retourne la lettre si c'est un candidat valide, None sinon.

#     Exemples gérés :
#       "A"                    → "A"   ✓
#       "Answer: B"            → "B"   ✓
#       "The answer is C."     → "C"   ✓
#       "Neither A nor B"      → None  ✓  (était mal parsé avant)
#       "F"  (= none_letter)   → None  ✓
#       ""                     → None  ✓
#     """
#     valid_letters = [chr(65 + i) for i in range(n_candidates)]

#     # Cherche une lettre isolée au début, ou après "answer" / ":"
#     patterns = [
#         r"^([A-Z])\b",                        # lettre en début de réponse
#         r"(?:answer[:\s]+)([A-Z])\b",         # "Answer: X" ou "answer X"
#         r"(?:is\s+)([A-Z])\b",                # "is X"
#         r"^\s*\**([A-Z])\**\s*$",             # "*A*" ou "**A**"
#     ]
#     for pattern in patterns:
#         m = re.search(pattern, answer.strip(), re.IGNORECASE)
#         if m:
#             letter = m.group(1).upper()
#             if letter in valid_letters:
#                 return letter
#             # C'est la lettre none_letter → pas de match
#             return None

#     return None


# def canonicalize_with_llm(client: OpenAI,
#                            triplet_uid: str,
#                            sentence: str,
#                            subject: str,
#                            predicate: str,
#                            obj: str,
#                            predicate_def: str,
#                            candidates: list,
#                            phase3_log: Path):
#     """
#     Phase 3 : MCQ LLM pour choisir la relation canonique.
#     Sauvegarde immédiatement dans phase3_log.

#     Retourne le nom de la relation choisie, ou None (= "None of the above").
#     """
#     choices_str, none_letter = build_choices(candidates)
#     prompt = CANONICALIZATION_PROMPT.format(
#         sentence=sentence,
#         subject=subject,
#         predicate=predicate,
#         object=obj,
#         predicate_def=predicate_def,
#         choices=choices_str,
#         none_letter=none_letter,
#     )
#     answer, meta = llm_call(client, prompt, max_tokens=10)

#     # ── Parsing robuste ───────────────────────────────────────────────────
#     chosen_letter   = parse_mcq_answer(answer, none_letter, len(candidates))
#     chosen_relation = None
#     if chosen_letter is not None:
#         idx = ord(chosen_letter) - 65
#         if 0 <= idx < len(candidates):
#             chosen_relation = candidates[idx][0]

#     record = {
#         "triplet_id":      triplet_uid,
#         "phase":           3,
#         "predicate":       predicate,
#         "sentence":        sentence,
#         "subject":         subject,
#         "object":          obj,
#         "predicate_def":   predicate_def,
#         "candidates":      [{"relation": r, "def": d} for r, d in candidates],
#         "none_letter":     none_letter,
#         "raw_answer":      answer,
#         "chosen_letter":   chosen_letter,
#         "chosen_relation": chosen_relation,
#         **meta,
#     }
#     append_jsonl(phase3_log, record)
#     log.debug("  [P3] '%s' → '%s' (lettre %s)",
#               predicate, chosen_relation or "None", chosen_letter)
#     return chosen_relation

# # ─── Embedding helpers ────────────────────────────────────────────────────────

# def embed(model: SentenceTransformer, texts: list) -> np.ndarray:
#     return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


# def top_k_by_similarity(query_vec: np.ndarray,
#                          corpus_vecs: np.ndarray,
#                          corpus_labels: list,
#                          k: int) -> list:
#     sims    = cosine_similarity(query_vec.reshape(1, -1), corpus_vecs)[0]
#     top_idx = np.argsort(sims)[::-1][:k]
#     return [(corpus_labels[i], float(sims[i])) for i in top_idx]

# # ─── Checkpoint ───────────────────────────────────────────────────────────────

# class CheckpointManager:
#     """
#     Persiste chaque triplet finalisé dans un JSONL.
#     Permet de reprendre le script sans repasser par l'API pour les triplets
#     déjà traités.
#     """

#     def __init__(self, path: Path):
#         self.path  = path
#         self._done = {}
#         self._load()

#     def _load(self):
#         for rec in load_jsonl(self.path):
#             tid = rec.get("triplet_id")
#             if tid:
#                 self._done[tid] = rec
#         if self._done:
#             log.info("Checkpoint : %d triplets rechargés depuis %s",
#                      len(self._done), self.path)

#     def is_done(self, tid: str) -> bool:
#         return tid in self._done

#     def get(self, tid: str) -> dict:
#         return self._done[tid]

#     def save(self, result: dict):
#         tid = result["triplet_id"]
#         self._done[tid] = result
#         append_jsonl(self.path, result)

#     def __len__(self):
#         return len(self._done)

# # ─── Pipeline principal ───────────────────────────────────────────────────────

# def run_canonicalization(ontology: dict,
#                          kg: list,
#                          client: OpenAI,
#                          output_path: Path,
#                          top_k: int = 5,
#                          embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
#                          ) -> list:
#     """
#     Phases 2 + 3 de l'article EDC — sans cache de définitions.

#     Changements clés vs. version avec cache :
#       1. Pas de DefinitionCache → Phase 2 appelée pour chaque triplet
#       2. Court-circuit si le prédicat est déjà dans l'ontologie (fidèle
#          au code officiel SchemaCanonicalizer.canonicalize)
#       3. Parsing MCQ robuste via parse_mcq_answer()
#     """

#     stem       = output_path.stem
#     parent     = output_path.parent
#     phase2_log = parent / f"{stem}.phase2.jsonl"
#     phase3_log = parent / f"{stem}.phase3.jsonl"
#     chk_path   = parent / f"{stem}.checkpoint.jsonl"

#     log.info("Fichiers de log :")
#     log.info("  Phase 2    → %s", phase2_log)
#     log.info("  Phase 3    → %s", phase3_log)
#     log.info("  Checkpoint → %s", chk_path)

#     checkpoint = CheckpointManager(chk_path)

#     # ── Ontologie ─────────────────────────────────────────────────────────
#     onto_names = list(ontology.keys())
#     onto_defs  = [ontology[r]["def"] for r in onto_names]
#     onto_set   = set(onto_names)          # pour le court-circuit O(1)
#     log.info("Ontologie : %d relations", len(onto_names))

#     # ── Embeddings de l'ontologie (calculés une seule fois) ───────────────
#     log.info("Calcul des embeddings ontologie (modèle : %s)…", embed_model_name)
#     embed_model = SentenceTransformer(embed_model_name)
#     onto_vecs   = embed(embed_model, onto_defs)
#     log.info("Embeddings shape : %s", onto_vecs.shape)

#     # ── Boucle principale ─────────────────────────────────────────────────
#     results        = []
#     skipped        = 0
#     already_canon  = 0
#     phase2_calls   = 0
#     phase3_calls   = 0

#     for triplet in tqdm(kg, desc="Canonicalisation"):
#         sentence  = triplet.get("sentence",  "")
#         subject   = triplet.get("subject",   "")
#         predicate = triplet.get("predicate", "")
#         obj       = triplet.get("object",    "")
#         tid       = triplet_id(triplet)

#         # ── 1. Triplet déjà traité (checkpoint) ──────────────────────────
#         if checkpoint.is_done(tid):
#             results.append(checkpoint.get(tid))
#             skipped += 1
#             continue

#         # ── 2. Court-circuit : prédicat déjà canonique ───────────────────
#         # Fidèle à SchemaCanonicalizer.canonicalize() du code officiel :
#         # "if open_relation in self.schema_dict: return open_triplet"
#         if predicate in onto_set:
#             result = {
#                 "triplet_id":           tid,
#                 **triplet,
#                 "original_predicate":   predicate,
#                 "predicate_definition": "(already canonical — LLM not called)",
#                 "canonical_predicate":  predicate,
#                 "canonicalized":        False,   # pas de transformation
#                 "top_candidates":       [],
#                 "processed_at":         now_iso(),
#             }
#             checkpoint.save(result)
#             results.append(result)
#             already_canon += 1
#             continue

#         # ── 3. Phase 2 : définition contextuelle du prédicat ─────────────
#         # Pas de cache → appel LLM systématique pour capturer le contexte
#         pred_def = generate_predicate_definition(
#             client, tid, sentence, subject, predicate, obj, phase2_log
#         )
#         phase2_calls += 1

#         # ── 4. Phase 3a : top-k candidats par similarité vectorielle ──────
#         query_vec      = embed(embed_model, [pred_def])[0]
#         top_candidates = top_k_by_similarity(
#             query_vec, onto_vecs, onto_names, k=top_k
#         )
#         candidates_for_llm = [
#             (rel, ontology[rel]["def"]) for rel, _score in top_candidates
#         ]

#         # ── 5. Phase 3b : vérification LLM (MCQ) ─────────────────────────
#         canonical = canonicalize_with_llm(
#             client, tid,
#             sentence, subject, predicate, obj,
#             pred_def, candidates_for_llm, phase3_log,
#         )
#         phase3_calls += 1

#         # ── 6. Résultat finalisé ──────────────────────────────────────────
#         result = {
#             "triplet_id":           tid,
#             **triplet,
#             "original_predicate":   predicate,
#             "predicate_definition": pred_def,
#             "canonical_predicate":  canonical if canonical else predicate,
#             "canonicalized":        canonical is not None,
#             "top_candidates": [
#                 {"relation": r, "similarity": round(s, 4)}
#                 for r, s in top_candidates
#             ],
#             "processed_at": now_iso(),
#         }

#         checkpoint.save(result)
#         results.append(result)

#     # ── Rapport final ─────────────────────────────────────────────────────
#     n_canon = sum(1 for r in results if r.get("canonicalized"))
#     log.info("─" * 60)
#     log.info("Triplets total             : %d", len(results))
#     log.info("  repris (checkpoint)      : %d", skipped)
#     log.info("  déjà canoniques (skip)   : %d", already_canon)
#     log.info("  traités par LLM          : %d", phase2_calls)
#     log.info("Canonicalisés              : %d / %d (%.1f%%)",
#              n_canon, len(results), 100 * n_canon / max(len(results), 1))
#     log.info("Appels LLM Phase 2         : %d", phase2_calls)
#     log.info("Appels LLM Phase 3         : %d", phase3_calls)
#     log.info("─" * 60)
#     return results

# # ─── Entry point ─────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="EDC canonicalization sans cache (DeepSeek)"
#     )
#     parser.add_argument("--ontology",    required=True,
#                         help="JSON ontologie {relation: {def:...}}")
#     parser.add_argument("--kg",          required=True,
#                         help="JSON KG [{sentence, subject, predicate, object, ...}]")
#     parser.add_argument("--output",      default="canonicalized.json",
#                         help="Fichier JSON de sortie final (default: canonicalized.json)")
#     parser.add_argument("--api_key",     required=True,
#                         help="Clé API DeepSeek")
#     parser.add_argument("--top_k",       type=int, default=5,
#                         help="Nombre de candidats pour le MCQ LLM (default: 5)")
#     parser.add_argument("--embed_model",
#                         default="sentence-transformers/all-MiniLM-L6-v2",
#                         help="Modèle sentence-transformers (default: all-MiniLM-L6-v2)")
#     parser.add_argument("--verbose",     action="store_true",
#                         help="Active les logs DEBUG")
#     args = parser.parse_args()

#     if args.verbose:
#         logging.getLogger().setLevel(logging.DEBUG)

#     with open(args.ontology, encoding="utf-8") as f:
#         ontology = json.load(f)

#     with open(args.kg, encoding="utf-8") as f:
#         kg_raw = json.load(f)

#     # Normaliser le KG en liste de triplets
#     if isinstance(kg_raw, list):
#         kg = kg_raw
#     elif isinstance(kg_raw, dict):
#         kg = []
#         for v in kg_raw.values():
#             if isinstance(v, list):
#                 for item in v:
#                     if isinstance(item, dict):
#                         kg.append(item)
#             elif isinstance(v, dict) and "def" not in v:
#                 kg.append(v)
#     log.info("KG normalisé : %d triplets", len(kg))

#     output_path = Path(args.output)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     client = make_client(args.api_key)

#     results = run_canonicalization(
#         ontology=ontology,
#         kg=kg,
#         client=client,
#         output_path=output_path,
#         top_k=args.top_k,
#         embed_model_name=args.embed_model,
#     )

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#     log.info("Résultats finaux sauvegardés → %s", output_path)


# if __name__ == "__main__":
#     main()


"""
EDC-style Schema Canonicalization — sans cache, avec multithreading
Follows the Extract-Define-Canonicalize paper (Zhang & Soh, EMNLP 2024)

Les appels LLM (Phase 2 + Phase 3) sont les goulots d'étranglement car ils
attendent la réponse réseau. On les parallélise avec ThreadPoolExecutor :
chaque triplet est traité dans un thread indépendant.

Précautions thread-safety :
  - CheckpointManager  : protégé par threading.Lock
  - append_jsonl       : protégé par un Lock dédié par fichier
  - SentenceTransformer encode() : appelé en batch avant les threads
    (les embeddings ontologie sont pré-calculés, read-only pendant les threads)
  - L'embedding de la définition (query_vec) est local à chaque thread

Usage:
  python canonicalize_edc_mt.py \\
      --ontology  ontology.json \\
      --kg        kg_triplets.json \\
      --output    canonicalized.json \\
      --api_key   sk-... \\
      --top_k     5 \\
      --workers   8
"""

import json
import re
import argparse
import logging
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ─── Utilitaires I/O thread-safe ──────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Un verrou par chemin de fichier pour éviter les écritures simultanées
_file_locks: dict[Path, threading.Lock] = {}
_file_locks_meta = threading.Lock()


def _get_file_lock(path: Path) -> threading.Lock:
    with _file_locks_meta:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


def append_jsonl(path: Path, record: dict):
    """Écriture thread-safe dans un fichier JSONL."""
    lock = _get_file_lock(path)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list:
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
    Appel API avec back-off exponentiel.
    Thread-safe : chaque thread a son propre stack d'appel, OpenAI client
    est thread-safe en lecture (pas d'état mutable partagé).
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
        "thread":     threading.current_thread().name,
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
            log.warning("[%s] API error (attempt %d/%d, attente %ds): %s",
                        threading.current_thread().name, attempt + 1, retries, wait, exc)
            meta["error"] = str(exc)
            time.sleep(wait)

    meta["response"] = ""
    return "", meta

# ─── Phase 2 — Schema Definition ──────────────────────────────────────────────

DEFINITION_PROMPT = """\
Given a piece of text and a relational triplet extracted from it, write a \
concise one-sentence definition for the relation (predicate) used in the triplet.

The definition must describe the *semantic role* of the relation in THIS \
specific context, not just paraphrase the sentence.

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
    prompt = DEFINITION_PROMPT.format(
        sentence=sentence, subject=subject, predicate=predicate, object=obj,
    )
    definition, meta = llm_call(client, prompt, max_tokens=150)
    record = {
        "triplet_id": triplet_uid, "phase": 2, "predicate": predicate,
        "sentence": sentence, "subject": subject, "object": obj, **meta,
    }
    append_jsonl(phase2_log, record)   # thread-safe via lock
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
    lines = []
    for i, (rel, rel_def) in enumerate(candidates):
        letter = chr(65 + i)
        lines.append(f"{letter}. '{rel}': {rel_def}")
    none_letter = chr(65 + len(candidates))
    lines.append(f"{none_letter}. None of the above")
    return "\n".join(lines), none_letter


def parse_mcq_answer(answer: str, none_letter: str, n_candidates: int) -> str | None:
    valid_letters = [chr(65 + i) for i in range(n_candidates)]
    patterns = [
        r"^([A-Z])\b",
        r"(?:answer[:\s]+)([A-Z])\b",
        r"(?:is\s+)([A-Z])\b",
        r"^\s*\**([A-Z])\**\s*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, answer.strip(), re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in valid_letters:
                return letter
            return None
    return None


def canonicalize_with_llm(client: OpenAI,
                           triplet_uid: str,
                           sentence: str,
                           subject: str,
                           predicate: str,
                           obj: str,
                           predicate_def: str,
                           candidates: list,
                           phase3_log: Path):
    choices_str, none_letter = build_choices(candidates)
    prompt = CANONICALIZATION_PROMPT.format(
        sentence=sentence, subject=subject, predicate=predicate, object=obj,
        predicate_def=predicate_def, choices=choices_str, none_letter=none_letter,
    )
    answer, meta = llm_call(client, prompt, max_tokens=10)

    chosen_letter = parse_mcq_answer(answer, none_letter, len(candidates))
    chosen_relation = None
    if chosen_letter is not None:
        idx = ord(chosen_letter) - 65
        if 0 <= idx < len(candidates):
            chosen_relation = candidates[idx][0]

    record = {
        "triplet_id": triplet_uid, "phase": 3, "predicate": predicate,
        "sentence": sentence, "subject": subject, "object": obj,
        "predicate_def": predicate_def,
        "candidates": [{"relation": r, "def": d} for r, d in candidates],
        "none_letter": none_letter, "raw_answer": answer,
        "chosen_letter": chosen_letter, "chosen_relation": chosen_relation,
        **meta,
    }
    append_jsonl(phase3_log, record)   # thread-safe via lock
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

# ─── Checkpoint thread-safe ───────────────────────────────────────────────────

class CheckpointManager:
    """
    Lecture/écriture protégée par un Lock.
    Plusieurs threads peuvent appeler is_done() / save() simultanément.
    """

    def __init__(self, path: Path):
        self.path  = path
        self._lock = threading.Lock()
        self._done: dict[str, dict] = {}
        self._load()

    def _load(self):
        for rec in load_jsonl(self.path):
            tid = rec.get("triplet_id")
            if tid:
                self._done[tid] = rec
        if self._done:
            log.info("Checkpoint : %d triplets rechargés", len(self._done))

    def is_done(self, tid: str) -> bool:
        with self._lock:
            return tid in self._done

    def get(self, tid: str) -> dict:
        with self._lock:
            return self._done[tid]

    def save(self, result: dict):
        tid = result["triplet_id"]
        with self._lock:
            self._done[tid] = result
        append_jsonl(self.path, result)   # append_jsonl a son propre lock

    def __len__(self):
        with self._lock:
            return len(self._done)

# ─── Traitement d'un triplet (exécuté dans un thread) ────────────────────────

def process_triplet(triplet: dict,
                    client: OpenAI,
                    embed_model: SentenceTransformer,
                    onto_names: list,
                    onto_defs: list,
                    onto_vecs: np.ndarray,
                    onto_set: set,
                    ontology: dict,
                    top_k: int,
                    phase2_log: Path,
                    phase3_log: Path,
                    checkpoint: CheckpointManager) -> dict:
    """
    Traite un triplet complet (Phase 2 + Phase 3).
    Conçu pour être appelé depuis un thread — toutes les variables partagées
    sont soit read-only (onto_vecs, onto_names, ontology), soit protégées
    (checkpoint, append_jsonl).
    """
    sentence  = triplet.get("sentence",  "")
    subject   = triplet.get("subject",   "")
    predicate = triplet.get("predicate", "")
    obj       = triplet.get("object",    "")
    tid       = triplet_id(triplet)

    # ── 1. Déjà traité ? ─────────────────────────────────────────────────
    if checkpoint.is_done(tid):
        return checkpoint.get(tid)

    # ── 2. Prédicat déjà canonique → court-circuit ───────────────────────
    if predicate in onto_set:
        result = {
            "triplet_id":           tid,
            **triplet,
            "original_predicate":   predicate,
            "predicate_definition": "(already canonical — LLM not called)",
            "canonical_predicate":  predicate,
            "canonicalized":        False,
            "top_candidates":       [],
            "processed_at":         now_iso(),
        }
        checkpoint.save(result)
        return result

    # ── 3. Phase 2 : définition contextuelle ────────────────────────────
    pred_def = generate_predicate_definition(
        client, tid, sentence, subject, predicate, obj, phase2_log
    )

    # ── 4. Phase 3a : top-k par similarité ───────────────────────────────
    # embed() ici est local au thread (numpy ops sont thread-safe en lecture)
    query_vec      = embed(embed_model, [pred_def])[0]
    top_candidates = top_k_by_similarity(query_vec, onto_vecs, onto_names, k=top_k)
    candidates_for_llm = [
        (rel, ontology[rel]["def"]) for rel, _score in top_candidates
    ]

    # ── 5. Phase 3b : MCQ LLM ────────────────────────────────────────────
    canonical = canonicalize_with_llm(
        client, tid, sentence, subject, predicate, obj,
        pred_def, candidates_for_llm, phase3_log,
    )

    # ── 6. Résultat ───────────────────────────────────────────────────────
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
    checkpoint.save(result)
    return result

# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_canonicalization(ontology: dict,
                         kg: list,
                         client: OpenAI,
                         output_path: Path,
                         top_k: int = 5,
                         embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                         max_workers: int = 8) -> list:
    """
    Phases 2 + 3 — multithreading.

    max_workers contrôle le nombre de triplets traités en parallèle.
    Chaque worker attend la réponse de l'API → I/O-bound → threads adaptés.

    Conseil : commencer avec 4-8 workers et augmenter selon les rate limits
    de votre API (DeepSeek limite les requêtes par minute).
    """

    stem       = output_path.stem
    parent     = output_path.parent
    phase2_log = parent / f"{stem}.phase2.jsonl"
    phase3_log = parent / f"{stem}.phase3.jsonl"
    chk_path   = parent / f"{stem}.checkpoint.jsonl"

    log.info("Workers : %d", max_workers)

    checkpoint = CheckpointManager(chk_path)

    # ── Ontologie + embeddings (séquentiel, fait une seule fois) ─────────
    onto_names = list(ontology.keys())
    onto_defs  = [ontology[r]["def"] for r in onto_names]
    onto_set   = set(onto_names)
    log.info("Ontologie : %d relations", len(onto_names))

    log.info("Calcul des embeddings ontologie…")
    embed_model = SentenceTransformer(embed_model_name)
    onto_vecs   = embed(embed_model, onto_defs)   # shape (n_onto, dim) — read-only ensuite
    log.info("Embeddings shape : %s", onto_vecs.shape)

    # ── Soumission des tâches au pool ────────────────────────────────────
    results_map: dict[str, dict] = {}   # tid → result, pour réordonner
    futures     = {}

    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix="edc") as pool:

        pbar = tqdm(total=len(kg), desc="Soumission")

        for triplet in kg:
            tid = triplet_id(triplet)

            # Si déjà dans le checkpoint, on récupère directement sans thread
            if checkpoint.is_done(tid):
                results_map[tid] = checkpoint.get(tid)
                pbar.update(1)
                continue

            future = pool.submit(
                process_triplet,
                triplet, client, embed_model,
                onto_names, onto_defs, onto_vecs, onto_set,
                ontology, top_k, phase2_log, phase3_log, checkpoint,
            )
            futures[future] = tid
            pbar.update(1)

        pbar.close()

        # ── Collecte des résultats au fur et à mesure ────────────────────
        pbar2 = tqdm(total=len(futures), desc="Complétion")
        for future in as_completed(futures):
            tid = futures[future]
            try:
                result = future.result()
                results_map[tid] = result
            except Exception as exc:
                log.error("Erreur sur triplet %s : %s", tid, exc)
            pbar2.update(1)
        pbar2.close()

    # ── Réordonner dans l'ordre original du KG ───────────────────────────
    results = []
    for triplet in kg:
        tid = triplet_id(triplet)
        if tid in results_map:
            results.append(results_map[tid])

    # ── Rapport ───────────────────────────────────────────────────────────
    n_canon       = sum(1 for r in results if r.get("canonicalized"))
    n_already     = sum(1 for r in results if r.get("predicate_definition", "").startswith("(already"))
    n_llm_called  = len(results) - n_already - sum(1 for r in results if checkpoint.is_done(triplet_id(r)) and r in [checkpoint.get(triplet_id(r))])

    log.info("─" * 60)
    log.info("Triplets total             : %d", len(results))
    log.info("  déjà canoniques (skip)   : %d", n_already)
    log.info("Canonicalisés              : %d / %d (%.1f%%)",
             n_canon, len(results), 100 * n_canon / max(len(results), 1))
    log.info("Workers utilisés           : %d", max_workers)
    log.info("─" * 60)
    return results

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDC canonicalization — sans cache, multithreading"
    )
    parser.add_argument("--ontology",  required=True)
    parser.add_argument("--kg",        required=True)
    parser.add_argument("--output",    default="canonicalized.json")
    parser.add_argument("--api_key",   required=True)
    parser.add_argument("--top_k",     type=int, default=5)
    parser.add_argument("--workers",   type=int, default=8,
                        help="Nombre de threads parallèles (default: 8)")
    parser.add_argument("--embed_model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    with open(args.ontology, encoding="utf-8") as f:
        ontology = json.load(f)

    with open(args.kg, encoding="utf-8") as f:
        kg_raw = json.load(f)

    kg = kg_raw if isinstance(kg_raw, list) else [
        item
        for v in kg_raw.values()
        for item in (v if isinstance(v, list) else [v])
        if isinstance(item, dict) and "def" not in item
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = make_client(args.api_key)

    results = run_canonicalization(
        ontology=ontology,
        kg=kg,
        client=client,
        output_path=output_path,
        top_k=args.top_k,
        embed_model_name=args.embed_model,
        max_workers=args.workers,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Résultats sauvegardés → %s", output_path)


if __name__ == "__main__":
    main()