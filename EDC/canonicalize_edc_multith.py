"""
EDC-style Schema Canonicalization — multithreadé + sauvegarde incrémentale
Follows the Extract-Define-Canonicalize paper (Zhang & Soh, EMNLP 2024)

Speedup via ThreadPoolExecutor : les appels API (I/O-bound) tournent en
parallèle. Les embeddings sont précalculés en batch avant la boucle.
Toutes les structures partagées sont protégées par des locks.

Fichiers produits
─────────────────
  <o>                   → résultats finaux JSON
  <o>.phase2.jsonl      → réponses brutes Phase 2 (prompt + réponse LLM)
  <o>.phase3.jsonl      → réponses brutes Phase 3 (prompt + réponse LLM)
  <o>.checkpoint.jsonl  → résultats finalisés (reprise en cas de crash)

Usage:
  python canonicalize_edc.py \\
      --ontology  ontology.json \\
      --kg        kg_triplets.json \\
      --output    canonicalized.json \\
      --api_key   sk-... \\
      --workers   10 \\
      --top_k     5
"""

import json
import argparse
import logging
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─── Logging : uniquement warnings/erreurs, tqdm propre ───────────────────────
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
for _lib in ("httpx", "httpcore", "openai", "sentence_transformers", "transformers"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# ─── Utilitaires I/O ──────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                    pass
    return records


def triplet_id(triplet: dict) -> str:
    key = json.dumps(
        {k: triplet.get(k, "") for k in ("sentence", "subject", "predicate", "object")},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]

# ─── Thread-safe file writer ──────────────────────────────────────────────────

class SafeWriter:
    """Écriture JSONL thread-safe : un lock par fichier."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

# ─── Thread-safe checkpoint ───────────────────────────────────────────────────

class CheckpointManager:
    """Persiste chaque triplet finalisé. Thread-safe."""

    def __init__(self, path: Path, writer: SafeWriter):
        self._writer = writer
        self._done: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load(path)

    def _load(self, path: Path):
        for rec in load_jsonl(path):
            tid = rec.get("triplet_id")
            if tid:
                self._done[tid] = rec

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
        self._writer.append(result)

    def __len__(self):
        with self._lock:
            return len(self._done)

# ─── Thread-safe definition cache ─────────────────────────────────────────────

class DefinitionCache:
    """
    Cache des définitions Phase 2, thread-safe.
    Un mécanisme 'pending' évite les appels API dupliqués pour le même prédicat
    quand deux threads le demandent simultanément.
    """

    def __init__(self):
        self._cache:   dict[str, str]   = {}
        self._pending: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def seed(self, path: Path):
        """Pré-charger depuis le log Phase 2 existant."""
        for rec in load_jsonl(path):
            pred = rec.get("predicate", "").lower().strip()
            defn = rec.get("response", "")
            if pred and defn:
                self._cache.setdefault(pred, defn)

    def get_or_reserve(self, predicate: str) -> tuple:
        """
        Retourne (définition, None)   si déjà en cache.
        Retourne (None, event)        si ce thread doit faire l'appel API
                                      (il doit appeler .publish() ensuite).
        Retourne (définition, None)   après attente si un autre thread
                                      faisait déjà l'appel.
        """
        key = predicate.lower().strip()
        with self._lock:
            if key in self._cache:
                return self._cache[key], None
            if key in self._pending:
                event = self._pending[key]
            else:
                event = threading.Event()
                self._pending[key] = event
                return None, event          # ce thread fait l'appel
        # Attendre que l'autre thread publie
        event.wait()
        with self._lock:
            return self._cache.get(key, ""), None

    def publish(self, predicate: str, definition: str):
        """Enregistrer la définition et débloquer les threads en attente."""
        key = predicate.lower().strip()
        with self._lock:
            self._cache[key] = definition
            event = self._pending.pop(key, None)
        if event:
            event.set()

    def __len__(self):
        with self._lock:
            return len(self._cache)

# ─── DeepSeek client ──────────────────────────────────────────────────────────

def make_client(api_key: str) -> OpenAI:
    """Le client OpenAI est thread-safe : un seul suffit pour tous les threads."""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def llm_call(client: OpenAI, prompt: str, max_tokens: int = 256,
             retries: int = 4) -> tuple:
    meta = {
        "prompt": prompt, "model": "deepseek-chat",
        "max_tokens": max_tokens, "called_at": now_iso(),
        "response": None, "usage": None, "error": None, "attempts": 0,
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
            meta["error"] = str(exc)
            time.sleep(wait)
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


def phase2_define(client: OpenAI,
                  tid: str,
                  sentence: str, subject: str, predicate: str, obj: str,
                  def_cache: DefinitionCache,
                  writer2: SafeWriter) -> str:
    """
    Retourne la définition du prédicat.
    Si déjà en cache → retour immédiat (pas d'appel API).
    Si un autre thread est en train de la calculer → attend.
    Sinon → appel API, publication dans le cache, écriture dans phase2 log.
    """
    definition, event = def_cache.get_or_reserve(predicate)

    if event is None:
        # Cache hit (ou résultat d'un autre thread)
        return definition

    # Ce thread est responsable de l'appel API
    prompt = DEFINITION_PROMPT.format(
        sentence=sentence, subject=subject, predicate=predicate, object=obj
    )
    definition, meta = llm_call(client, prompt, max_tokens=150)

    writer2.append({
        "triplet_id": tid, "phase": 2,
        "predicate": predicate, "sentence": sentence,
        "subject": subject, "object": obj,
        **meta,
    })

    def_cache.publish(predicate, definition)
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
        lines.append(f"{chr(65+i)}. '{rel}': {rel_def}")
    none_letter = chr(65 + len(candidates))
    lines.append(f"{none_letter}. None of the above")
    return "\n".join(lines), none_letter


def phase3_canonicalize(client: OpenAI,
                         tid: str,
                         sentence: str, subject: str, predicate: str, obj: str,
                         predicate_def: str,
                         candidates: list,
                         writer3: SafeWriter):
    choices_str, none_letter = build_choices(candidates)
    prompt = CANONICALIZATION_PROMPT.format(
        sentence=sentence, subject=subject, predicate=predicate, object=obj,
        predicate_def=predicate_def, choices=choices_str, none_letter=none_letter,
    )
    answer, meta = llm_call(client, prompt, max_tokens=10)
    answer_upper = answer.upper()

    chosen_relation = None
    chosen_letter   = none_letter
    for i, (rel, _) in enumerate(candidates):
        if chr(65 + i) in answer_upper:
            chosen_relation = rel
            chosen_letter   = chr(65 + i)
            break

    writer3.append({
        "triplet_id": tid, "phase": 3,
        "predicate": predicate, "sentence": sentence,
        "subject": subject, "object": obj,
        "predicate_def": predicate_def,
        "candidates": [{"relation": r, "def": d} for r, d in candidates],
        "none_letter": none_letter,
        "chosen_letter": chosen_letter,
        "chosen_relation": chosen_relation,
        **meta,
    })
    return chosen_relation

# ─── Embedding helpers ────────────────────────────────────────────────────────

def batch_embed(model: SentenceTransformer, texts: list) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False,
                        convert_to_numpy=True, batch_size=256)


def top_k_by_similarity(query_vec: np.ndarray,
                         corpus_vecs: np.ndarray,
                         corpus_labels: list, k: int) -> list:
    sims    = cosine_similarity(query_vec.reshape(1, -1), corpus_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [(corpus_labels[i], float(sims[i])) for i in top_idx]

# ─── Worker : traite un seul triplet ─────────────────────────────────────────

def process_triplet(triplet: dict,
                    client: OpenAI,
                    onto_names: list,
                    onto_vecs: np.ndarray,
                    ontology: dict,
                    embed_model: SentenceTransformer,
                    def_cache: DefinitionCache,
                    checkpoint: CheckpointManager,
                    writer2: SafeWriter,
                    writer3: SafeWriter,
                    top_k: int) -> dict:
    """Exécuté dans un thread : Phase 2 → embed → Phase 3 → checkpoint."""

    sentence  = triplet.get("sentence",  "")
    subject   = triplet.get("subject",   "")
    predicate = triplet.get("predicate", "")
    obj       = triplet.get("object",    "")
    tid       = triplet_id(triplet)

    # ── Reprise ───────────────────────────────────────────────────────────
    if checkpoint.is_done(tid):
        return checkpoint.get(tid)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    pred_def = phase2_define(
        client, tid, sentence, subject, predicate, obj, def_cache, writer2
    )

    # ── Phase 3a : similarité vectorielle ─────────────────────────────────
    query_vec      = batch_embed(embed_model, [pred_def])[0]
    top_candidates = top_k_by_similarity(query_vec, onto_vecs, onto_names, top_k)
    candidates_llm = [(r, ontology[r]["def"]) for r, _ in top_candidates]

    # ── Phase 3b : vérification LLM ───────────────────────────────────────
    canonical = phase3_canonicalize(
        client, tid, sentence, subject, predicate, obj,
        pred_def, candidates_llm, writer3,
    )

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
                         workers: int = 10,
                         embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                         ) -> list:

    stem   = output_path.stem
    parent = output_path.parent
    phase2_log = parent / f"{stem}.phase2.jsonl"
    phase3_log = parent / f"{stem}.phase3.jsonl"
    chk_path   = parent / f"{stem}.checkpoint.jsonl"

    # ── Writers thread-safe ───────────────────────────────────────────────
    writer2    = SafeWriter(phase2_log)
    writer3    = SafeWriter(phase3_log)
    chk_writer = SafeWriter(chk_path)

    # ── Checkpoint + cache ────────────────────────────────────────────────
    checkpoint = CheckpointManager(chk_path, chk_writer)
    def_cache  = DefinitionCache()
    def_cache.seed(phase2_log)

    n_skip = len(checkpoint)
    if n_skip:
        tqdm.write(f"Reprise : {n_skip} triplets déjà traités ignorés.")

    # ── Ontologie + embeddings (batch, une seule fois) ────────────────────
    onto_names = list(ontology.keys())
    onto_defs  = [ontology[r]["def"] for r in onto_names]

    tqdm.write(f"Chargement du modèle d'embedding : {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)
    tqdm.write(f"Calcul des embeddings ({len(onto_names)} relations ontologie)…")
    onto_vecs = batch_embed(embed_model, onto_defs)
    tqdm.write(f"Prêt. Lancement avec {workers} workers parallèles.\n")

    # ── Filtrer les triplets déjà traités ─────────────────────────────────
    todo    = [t for t in kg if not checkpoint.is_done(triplet_id(t))]
    skipped = [checkpoint.get(triplet_id(t)) for t in kg
               if checkpoint.is_done(triplet_id(t))]

    # ── Exécution parallèle ───────────────────────────────────────────────
    results_map: dict[str, dict] = {r["triplet_id"]: r for r in skipped}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_triplet,
                triplet, client,
                onto_names, onto_vecs, ontology,
                embed_model, def_cache, checkpoint,
                writer2, writer3, top_k,
            ): triplet
            for triplet in todo
        }

        with tqdm(total=len(kg), initial=len(skipped),
                  desc="Canonicalisation", unit="triplet") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results_map[result["triplet_id"]] = result
                except Exception as exc:
                    t = futures[future]
                    tqdm.write(f"[ERROR] {triplet_id(t)} : {exc}")
                finally:
                    pbar.update(1)

    # ── Réordonner selon l'ordre d'entrée ─────────────────────────────────
    results = []
    for triplet in kg:
        tid = triplet_id(triplet)
        if tid in results_map:
            results.append(results_map[tid])

    # ── Stats ─────────────────────────────────────────────────────────────
    n_canon = sum(1 for r in results if r.get("canonicalized"))
    tqdm.write(
        f"\nTerminé : {len(results)} triplets | "
        f"{n_canon} canonicalisés ({100*n_canon/max(len(results),1):.1f}%) | "
        f"{len(skipped)} repris du checkpoint"
    )
    return results

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDC canonicalization — multithreadé (DeepSeek)"
    )
    parser.add_argument("--ontology",    required=True)
    parser.add_argument("--kg",          required=True)
    parser.add_argument("--output",      default="canonicalized.json")
    parser.add_argument("--api_key",     required=True)
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--workers",     type=int, default=10,
                        help="Threads parallèles (default: 10)")
    parser.add_argument("--embed_model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    with open(args.ontology, encoding="utf-8") as f:
        ontology = json.load(f)

    with open(args.kg, encoding="utf-8") as f:
        kg_raw = json.load(f)

    if isinstance(kg_raw, list):
        kg = kg_raw
    elif isinstance(kg_raw, dict):
        kg = []
        for v in kg_raw.values():
            if isinstance(v, list):
                kg += [item for item in v if isinstance(item, dict)]
            elif isinstance(v, dict) and "def" not in v:
                kg.append(v)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = make_client(args.api_key)

    results = run_canonicalization(
        ontology=ontology,
        kg=kg,
        client=client,
        output_path=output_path,
        top_k=args.top_k,
        workers=args.workers,
        embed_model_name=args.embed_model,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tqdm.write(f"Résultats sauvegardés → {output_path}")


if __name__ == "__main__":
    main()