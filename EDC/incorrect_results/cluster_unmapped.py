"""
Clustering & normalisation des prédicats non-mappés (canonicalized=false)
─────────────────────────────────────────────────────────────────────────
Pipeline :
  1. Extraire tous les prédicats non-mappés + leurs définitions Phase 2
  2. Embeddings → clustering hiérarchique agglomératif
  3. Pour chaque cluster → LLM choisit un label canonique parmi les membres
  4. Mise à jour du fichier de résultats EDC

Fichiers produits :
  <o>                      → JSON final mis à jour
  <o>.clusters.json        → clusters et leurs membres
  <o>.cluster_labels.jsonl → réponses brutes LLM pour chaque cluster

Usage :
  python cluster_unmapped.py \\
      --input   edc_canonicalized.json \\
      --output  edc_final.json \\
      --api_key sk-... \\
      --workers 8 \\
      --distance_threshold 0.35
"""

import json
import argparse
import logging
import time
import threading
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─── Logging silencieux ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for _lib in ("httpx", "httpcore", "openai", "sentence_transformers", "transformers"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# ─── Helpers I/O ──────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

class SafeWriter:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
    def append(self, record: dict):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

# ─── DeepSeek ────────────────────────────────────────────────────────────────

def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def llm_call(client: OpenAI, prompt: str, max_tokens: int = 100, retries: int = 4) -> tuple:
    meta = {
        "prompt": prompt, "model": "deepseek-chat",
        "called_at": now_iso(), "response": None,
        "usage": None, "error": None, "attempts": 0,
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
            meta["error"] = str(exc)
            time.sleep(2 ** attempt)
    meta["response"] = ""
    return "", meta

# ─── Prompt de labellisation d'un cluster ────────────────────────────────────

LABEL_PROMPT = """\
You are a knowledge graph expert. Below is a group of semantically related \
relation predicates extracted from biomedical text, along with their definitions.

Your task: choose ONE single canonical relation label that best represents \
ALL members of this group. Prefer concise snake_case labels (e.g. \
'has_finding', 'location_of', 'associated_with').

Members:
{members}

Rules:
- Output ONLY the canonical label, nothing else.
- The label must be a single relation phrase (no spaces, use underscores).
- Do not invent a completely new concept; prefer the most representative \
  member if it is already clean."""

def label_cluster(client: OpenAI,
                  cluster_id: int,
                  members: list,          # list of {predicate, definition}
                  writer: SafeWriter) -> str:
    """
    Demande au LLM un label canonique pour un cluster.
    members : [{"predicate": "...", "definition": "..."}, ...]
    """
    members_text = "\n".join(
        f"  - '{m['predicate']}': {m['definition']}"
        for m in members
    )
    prompt = LABEL_PROMPT.format(members=members_text)
    label, meta = llm_call(client, prompt, max_tokens=20)

    # Nettoyer : garder uniquement le premier mot/token snake_case
    label = label.strip().split()[0] if label.strip() else members[0]["predicate"]
    label = label.strip("'\".,;:")

    writer.append({
        "cluster_id":    cluster_id,
        "members":       [m["predicate"] for m in members],
        "chosen_label":  label,
        "called_at":     now_iso(),
        **meta,
    })
    return label

# ─── Embedding ───────────────────────────────────────────────────────────────

def batch_embed(model: SentenceTransformer, texts: list) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False,
                        convert_to_numpy=True, batch_size=256)

# ─── Étape 1 : extraire prédicats non-mappés ─────────────────────────────────

def extract_unmapped(records: list) -> dict:
    """
    Retourne {predicate_lower: {"predicate": ..., "definition": ...}}
    pour tous les triplets non canonicalisés.
    Déduplique par prédicat.
    """
    unmapped = {}
    for r in records:
        if not r.get("canonicalized", True):
            pred = r.get("original_predicate", r.get("predicate", "")).strip()
            defn = r.get("predicate_definition", "")
            key  = pred.lower()
            if key and key not in unmapped:
                unmapped[key] = {"predicate": pred, "definition": defn}
    return unmapped

# ─── Étape 2 : clustering agglomératif ───────────────────────────────────────

def cluster_predicates(unmapped: dict,
                       embed_model: SentenceTransformer,
                       distance_threshold: float) -> dict:
    """
    Retourne {cluster_id: [{"predicate": ..., "definition": ...}, ...]}
    """
    keys  = list(unmapped.keys())
    items = [unmapped[k] for k in keys]

    tqdm.write(f"  Embedding {len(items)} prédicats non-mappés…")

    # Utiliser predicate + definition pour l'embedding (plus riche)
    texts = [
        f"{it['predicate']}: {it['definition']}" if it['definition']
        else it['predicate']
        for it in items
    ]
    vecs = batch_embed(embed_model, texts)

    # Distance cosinus = 1 - similarité cosinus
    tqdm.write(f"  Clustering (distance_threshold={distance_threshold})…")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(vecs)

    clusters = defaultdict(list)
    for item, cluster_id in zip(items, labels):
        clusters[int(cluster_id)].append(item)

    tqdm.write(f"  → {len(clusters)} clusters formés pour {len(items)} prédicats")
    return dict(clusters)

# ─── Étape 3 : labelliser chaque cluster en parallèle ────────────────────────

def label_all_clusters(clusters: dict,
                       client: OpenAI,
                       writer: SafeWriter,
                       workers: int) -> dict:
    """
    Retourne {cluster_id: canonical_label}
    Les clusters à 1 membre ne nécessitent pas d'appel LLM.
    """
    labels = {}
    singles   = {cid: mems for cid, mems in clusters.items() if len(mems) == 1}
    multiples = {cid: mems for cid, mems in clusters.items() if len(mems) > 1}

    # Clusters singletons : label = le prédicat lui-même (nettoyé)
    for cid, mems in singles.items():
        pred = mems[0]["predicate"]
        # Normaliser : remplacer espaces et tirets par underscores, minuscules
        clean = pred.lower().replace(" ", "_").replace("-", "_")
        labels[cid] = clean

    tqdm.write(f"  {len(singles)} singletons (pas d'appel LLM)")
    tqdm.write(f"  {len(multiples)} clusters multi-membres → appels LLM…")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(label_cluster, client, cid, mems, writer): cid
            for cid, mems in multiples.items()
        }
        with tqdm(total=len(multiples), desc="Labellisation clusters", unit="cluster") as pbar:
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    labels[cid] = future.result()
                except Exception as exc:
                    tqdm.write(f"[ERROR] cluster {cid}: {exc}")
                    labels[cid] = clusters[cid][0]["predicate"]
                finally:
                    pbar.update(1)

    return labels

# ─── Étape 4 : construire le mapping predicate → canonical_label ──────────────

def build_predicate_mapping(clusters: dict, cluster_labels: dict) -> dict:
    """
    {predicate_lower: canonical_label}
    """
    mapping = {}
    for cid, members in clusters.items():
        label = cluster_labels[cid]
        for m in members:
            mapping[m["predicate"].lower()] = label
    return mapping

# ─── Étape 5 : mettre à jour les enregistrements EDC ─────────────────────────

def apply_mapping(records: list, predicate_mapping: dict) -> tuple:
    """
    Met à jour canonical_predicate pour les triplets non-mappés.
    Retourne (updated_records, n_updated).
    """
    n_updated = 0
    for r in records:
        if not r.get("canonicalized", True):
            pred_key = r.get("original_predicate", r.get("predicate", "")).lower()
            if pred_key in predicate_mapping:
                r["canonical_predicate"] = predicate_mapping[pred_key]
                r["canonicalized"]       = True
                r["canonicalization_method"] = "cluster_llm"
                n_updated += 1
    return records, n_updated

# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_clustering(input_path: Path,
                   output_path: Path,
                   client: OpenAI,
                   distance_threshold: float = 0.35,
                   workers: int = 8,
                   embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

    stem   = output_path.stem
    parent = output_path.parent
    clusters_path = parent / f"{stem}.clusters.json"
    labels_log    = parent / f"{stem}.cluster_labels.jsonl"
    writer        = SafeWriter(labels_log)

    # ── Charger les résultats EDC ─────────────────────────────────────────
    tqdm.write(f"Chargement : {input_path}")
    records = load_json(input_path)
    n_total    = len(records)
    n_unmapped = sum(1 for r in records if not r.get("canonicalized", True))
    n_mapped   = n_total - n_unmapped
    tqdm.write(f"  {n_total} triplets | {n_mapped} mappés | {n_unmapped} non-mappés")

    if n_unmapped == 0:
        tqdm.write("Aucun prédicat non-mappé. Rien à faire.")
        save_json(output_path, records)
        return

    # ── Étape 1 : extraire prédicats uniques non-mappés ───────────────────
    unmapped = extract_unmapped(records)
    tqdm.write(f"  {len(unmapped)} prédicats uniques non-mappés")

    # ── Étape 2 : embeddings + clustering ─────────────────────────────────
    tqdm.write(f"\nChargement modèle embedding : {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)
    clusters    = cluster_predicates(unmapped, embed_model, distance_threshold)

    # Sauvegarder les clusters
    save_json(clusters_path, {
        str(cid): [m["predicate"] for m in mems]
        for cid, mems in clusters.items()
    })
    tqdm.write(f"  Clusters sauvegardés → {clusters_path}")

    # ── Étape 3 : labelliser chaque cluster via LLM ───────────────────────
    tqdm.write(f"\nLabellisation LLM ({workers} workers)…")
    cluster_labels = label_all_clusters(clusters, client, writer, workers)

    # ── Étape 4 : mapping predicate → label ──────────────────────────────
    predicate_mapping = build_predicate_mapping(clusters, cluster_labels)

    # Afficher un aperçu
    tqdm.write("\nAperçu du mapping (20 premiers) :")
    for pred, label in list(predicate_mapping.items())[:20]:
        tqdm.write(f"  '{pred}' → '{label}'")

    # ── Étape 5 : mettre à jour les enregistrements ───────────────────────
    records, n_updated = apply_mapping(records, predicate_mapping)

    # ── Sauvegarder ───────────────────────────────────────────────────────
    save_json(output_path, records)

    # ── Stats finales ─────────────────────────────────────────────────────
    n_still_unmapped = sum(1 for r in records if not r.get("canonicalized", True))
    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"Triplets total             : {n_total}")
    tqdm.write(f"Mappés avant clustering    : {n_mapped}")
    tqdm.write(f"Mappés par clustering LLM  : {n_updated}")
    tqdm.write(f"Encore non-mappés          : {n_still_unmapped}")
    tqdm.write(f"Résultats finaux           → {output_path}")
    tqdm.write(f"Clusters                   → {clusters_path}")
    tqdm.write(f"Logs labels LLM            → {labels_log}")
    tqdm.write(f"{'─'*60}")

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clustering + normalisation LLM des prédicats non-mappés"
    )
    parser.add_argument("--input",    required=True,
                        help="JSON de sortie EDC (edc_canonicalized.json)")
    parser.add_argument("--output",   required=True,
                        help="JSON final après clustering (edc_final.json)")
    parser.add_argument("--api_key",  required=True)
    parser.add_argument("--workers",  type=int, default=8,
                        help="Threads parallèles pour les appels LLM (default: 8)")
    parser.add_argument("--distance_threshold", type=float, default=0.35,
                        help="Seuil de distance cosinus pour le clustering "
                             "(0.2=clusters très serrés, 0.5=très larges, default: 0.35)")
    parser.add_argument("--embed_model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = make_client(args.api_key)

    run_clustering(
        input_path=input_path,
        output_path=output_path,
        client=client,
        distance_threshold=args.distance_threshold,
        workers=args.workers,
        embed_model_name=args.embed_model,
    )

if __name__ == "__main__":
    main()
