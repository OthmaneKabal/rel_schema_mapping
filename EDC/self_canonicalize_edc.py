"""
EDC Self Canonicalization — fidèle à l'article (Zhang & Soh, EMNLP 2024)

Section 3.1 de l'article :
  "Absent a target schema, the goal is to consolidate semantically similar
   schema components, standardizing them to a singular representation.
   Starting with an empty canonical schema, we examine the open KG triplets,
   searching for potential consolidation candidates through vector similarity
   and LLM verification. Unlike target alignment, components deemed
   non-transformable are ADDED to the canonical schema, thereby expanding it."

Input attendu :
  Le fichier JSON produit par canonicalize_edc_mt.py filtré sur les triplets
  non mappés (canonicalized=False et pas déjà dans l'ontologie).
  Ces triplets ont déjà une predicate_definition de la Phase 2 — on la
  réutilise directement, pas besoin de rappeler le LLM.

Sortie :
  <output>                     → triplets avec canonical_predicate final
  <output>.schema.json         → schéma canonique auto-construit
  <output>.checkpoint.jsonl    → reprise possible

Usage:
  python self_canonicalize_edc.py \\
      --kg       non_mapped_triplets.json \\
      --output   self_canonicalized.json \\
      --api_key  sk-... \\
      --top_k    5 \\
      --verbose
"""

import json
import re
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

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for _lib in ("httpx", "httpcore", "openai", "sentence_transformers", "transformers"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# ─── Utilitaires ──────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict):
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
                    log.warning("Ligne JSONL corrompue ignorée : %s", path)
    return records


def triplet_id(triplet: dict) -> str:
    key = json.dumps(
        {k: triplet.get(k, "") for k in ("sentence", "subject", "predicate", "object")},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]

# ─── Client LLM ───────────────────────────────────────────────────────────────

def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def llm_call(client: OpenAI,
             prompt: str,
             max_tokens: int = 256,
             retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("API error (attempt %d/%d, attente %ds): %s",
                        attempt + 1, retries, wait, exc)
            time.sleep(wait)
    return ""

# ─── Phase 2 — Schema Definition (si définition manquante) ────────────────────
# Dans le cas normal, les triplets non-mappés ont déjà leur predicate_definition
# car ils ont traversé la Phase 2 dans canonicalize_edc_mt.py.
# On ne rappelle le LLM que si la définition est absente.

DEFINITION_PROMPT = """\
Given a piece of text and a relational triplet extracted from it, write a \
concise one-sentence definition for the relation (predicate) used in the triplet.

The definition must describe the *semantic role* of the relation in THIS \
specific context, not just paraphrase the sentence.

Text: {sentence}
Triplet: ['{subject}', '{predicate}', '{object}']

Write only the definition, nothing else."""


def get_definition(client: OpenAI, triplet: dict) -> str:
    """
    Retourne la définition existante si disponible (pas de rappel LLM),
    sinon génère une nouvelle définition contextuelle.
    """
    existing = triplet.get("predicate_definition", "").strip()

    # Définitions invalides laissées par la phase Target Alignment
    invalid = {"", "(already canonical — llm not called)"}
    if existing.lower() not in invalid:
        log.debug("  [P2] Définition réutilisée pour '%s'", triplet.get("predicate"))
        return existing

    # Rare : la définition est absente, on la génère
    log.debug("  [P2] Définition manquante pour '%s' → appel LLM", triplet.get("predicate"))
    prompt = DEFINITION_PROMPT.format(
        sentence=triplet.get("sentence",  ""),
        subject=triplet.get("subject",   ""),
        predicate=triplet.get("predicate", ""),
        object=triplet.get("object",    ""),
    )
    return llm_call(client, prompt, max_tokens=150)

# ─── Phase 3 — Self Canonicalization MCQ ──────────────────────────────────────

# Fidèle au prompt de l'article (Section 3.1, Schema Canonicalization Prompt)
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
    Retourne (texte_formaté, lettre_none)
    """
    lines = []
    for i, (rel, rel_def) in enumerate(candidates):
        letter = chr(65 + i)
        lines.append(f"{letter}. '{rel}': {rel_def}")
    none_letter = chr(65 + len(candidates))
    lines.append(f"{none_letter}. None of the above")
    return "\n".join(lines), none_letter


def parse_mcq_answer(answer: str, n_candidates: int) -> str | None:
    """Parsing robuste — cherche une lettre isolée parmi les candidats valides."""
    valid = [chr(65 + i) for i in range(n_candidates)]
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
            if letter in valid:
                return letter
            return None   # lettre none_letter ou invalide
    return None

# ─── Embedding helpers ────────────────────────────────────────────────────────

def embed_texts(model: SentenceTransformer, texts: list) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def top_k_from_schema(query_vec: np.ndarray,
                      schema_names: list,
                      schema_vecs: list,
                      k: int) -> list:
    """
    Retourne les top-k relations du schéma dynamique par similarité cosinus.
    schema_vecs : liste de vecteurs numpy (peut grandir au fil du traitement)
    """
    if not schema_vecs:
        return []
    corpus = np.stack(schema_vecs)
    sims   = cosine_similarity(query_vec.reshape(1, -1), corpus)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [(schema_names[i], float(sims[i])) for i in top_idx]

# ─── Schéma canonique dynamique ───────────────────────────────────────────────

class DynamicSchema:
    """
    Représente le schéma canonique auto-construit qui grandit au fil du
    traitement des triplets, exactement comme décrit dans l'article :

    "Starting with an empty canonical schema, we examine the open KG triplets,
     searching for potential consolidation candidates through vector similarity
     and LLM verification. Components deemed non-transformable are added to
     the canonical schema, thereby expanding it."
    """

    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model
        self.relations:   list[str]        = []   # noms des relations canoniques
        self.definitions: list[str]        = []   # leurs définitions
        self.embeddings:  list[np.ndarray] = []   # leurs vecteurs

    def __len__(self):
        return len(self.relations)

    def contains(self, relation: str) -> bool:
        return relation in self.relations

    def add(self, relation: str, definition: str):
        """
        Ajoute une nouvelle relation canonique au schéma.
        Appelé quand le LLM juge qu'aucun candidat existant ne convient.
        """
        vec = embed_texts(self.embed_model, [definition])[0]
        self.relations.append(relation)
        self.definitions.append(definition)
        self.embeddings.append(vec)
        log.info("  [SCHEMA] Nouvelle relation ajoutée : '%s'", relation)

    def find_candidates(self, query_vec: np.ndarray, k: int) -> list:
        """Retourne les top-k candidats [(relation, score), ...]."""
        return top_k_from_schema(query_vec, self.relations, self.embeddings, k)

    def to_dict(self) -> dict:
        return {
            rel: {"def": defn}
            for rel, defn in zip(self.relations, self.definitions)
        }

# ─── Checkpoint ───────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, path: Path):
        self.path  = path
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
        return tid in self._done

    def get(self, tid: str) -> dict:
        return self._done[tid]

    def save(self, result: dict):
        self._done[result["triplet_id"]] = result
        append_jsonl(self.path, result)

    def __len__(self):
        return len(self._done)

# ─── Pipeline Self Canonicalization ──────────────────────────────────────────

def run_self_canonicalization(kg: list,
                              client: OpenAI,
                              embed_model: SentenceTransformer,
                              output_path: Path,
                              top_k: int = 5) -> tuple:
    """
    Self Canonicalization fidèle à l'article — séquentiel.

    Pour chaque triplet :
      1. Récupère la définition (Phase 2 déjà faite ou nouveau appel LLM)
      2. Cherche des candidats dans le schéma dynamique (similarité cosinus)
      3. LLM vérifie si une fusion est possible (MCQ)
         - Oui → canonicalisé vers la relation existante
         - Non → relation ajoutée au schéma dynamique (expansion)

    Retourne (results, dynamic_schema)
    """
    stem     = output_path.stem
    parent   = output_path.parent
    chk_path = parent / f"{stem}.checkpoint.jsonl"

    checkpoint = CheckpointManager(chk_path)

    # Schéma canonique dynamique — commence vide
    schema = DynamicSchema(embed_model)
    log.info("Schéma dynamique initialisé (vide)")

    # ── Si on reprend depuis un checkpoint, reconstruire le schéma ────────
    # On rejoue les résultats déjà sauvegardés pour remettre le schéma dans
    # l'état cohérent avant de continuer
    if len(checkpoint) > 0:
        log.info("Reconstruction du schéma depuis le checkpoint…")
        for tid, rec in checkpoint._done.items():
            canon = rec.get("canonical_predicate", "")
            canon_def = rec.get("predicate_definition", "")
            if canon and not schema.contains(canon):
                # Seules les relations qui ont été ajoutées au schéma
                # (canonicalized=False = ajoutée) doivent être rechargées
                if not rec.get("canonicalized", False):
                    schema.add(canon, canon_def)
        log.info("Schéma reconstruit : %d relations", len(schema))

    results = []
    n_fused   = 0   # fusionnés vers une relation existante
    n_added   = 0   # ajoutés comme nouvelle relation canonique
    n_skipped = 0   # repris depuis checkpoint

    for triplet in tqdm(kg, desc="Self Canonicalization"):
        sentence  = triplet.get("sentence",  "")
        subject   = triplet.get("subject",   "")
        predicate = triplet.get("predicate", "")
        obj       = triplet.get("object",    "")
        tid       = triplet_id(triplet)

        # ── Checkpoint : triplet déjà traité ─────────────────────────────
        if checkpoint.is_done(tid):
            results.append(checkpoint.get(tid))
            n_skipped += 1
            continue

        # ── Phase 2 : définition contextuelle ────────────────────────────
        pred_def = get_definition(client, triplet)

        # ── Embedding de la définition ────────────────────────────────────
        query_vec = embed_texts(embed_model, [pred_def])[0]

        # ── Phase 3 : Self Canonicalization ──────────────────────────────
        #
        # Cas 1 : schéma encore vide → aucun candidat possible
        #         → on ajoute directement la relation au schéma
        if len(schema) == 0:
            schema.add(predicate, pred_def)
            canonical_predicate = predicate
            canonicalized       = False   # pas de fusion, c'est une fondation
            top_candidates      = []
            n_added += 1

        else:
            # Cas 2 : schéma non vide → on cherche des candidats similaires
            top_candidates = schema.find_candidates(query_vec, k=top_k)
            candidates_for_llm = [
                (rel, schema.definitions[schema.relations.index(rel)])
                for rel, _score in top_candidates
            ]

            # MCQ LLM
            choices_str, none_letter = build_choices(candidates_for_llm)
            prompt = CANONICALIZATION_PROMPT.format(
                sentence=sentence,
                subject=subject,
                predicate=predicate,
                object=obj,
                predicate_def=pred_def,
                choices=choices_str,
                none_letter=none_letter,
            )
            answer = llm_call(client, prompt, max_tokens=10)
            chosen_letter = parse_mcq_answer(answer, n_candidates=len(candidates_for_llm))

            if chosen_letter is not None:
                # ── LLM a choisi un candidat → fusion ────────────────────
                idx = ord(chosen_letter) - 65
                canonical_predicate = candidates_for_llm[idx][0]
                canonicalized       = True
                n_fused += 1
                log.debug("  [FUSED] '%s' → '%s'", predicate, canonical_predicate)

            else:
                # ── Aucun candidat acceptable → ajout au schéma ──────────
                # C'est le comportement clé de Self Canonicalization :
                # "components deemed non-transformable are added to the
                #  canonical schema, thereby expanding it."
                schema.add(predicate, pred_def)
                canonical_predicate = predicate
                canonicalized       = False
                n_added += 1
                log.debug("  [ADDED] '%s' ajouté au schéma", predicate)

        # ── Résultat finalisé ─────────────────────────────────────────────
        result = {
            "triplet_id":           tid,
            **{k: v for k, v in triplet.items() if k != "triplet_id"},
            "original_predicate":   predicate,
            "predicate_definition": pred_def,
            "canonical_predicate":  canonical_predicate,
            "canonicalized":        canonicalized,   # True = fusionné, False = ajouté
            "top_candidates": [
                {"relation": r, "similarity": round(s, 4)}
                for r, s in top_candidates
            ],
            "processed_at": now_iso(),
        }

        checkpoint.save(result)
        results.append(result)

    # ── Rapport ───────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Triplets traités           : %d", len(results))
    log.info("  repris (checkpoint)      : %d", n_skipped)
    log.info("  fusionnés (canonicalisés): %d", n_fused)
    log.info("  ajoutés au schéma        : %d", n_added)
    log.info("Schéma final               : %d relations distinctes", len(schema))
    log.info("─" * 60)

    return results, schema

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDC Self Canonicalization — fidèle à l'article"
    )
    parser.add_argument("--kg",       required=True,
                        help="JSON des triplets non mappés (output de canonicalize_edc_mt.py filtré)")
    parser.add_argument("--output",   default="self_canonicalized.json")
    parser.add_argument("--api_key",  required=True)
    parser.add_argument("--top_k",    type=int, default=5,
                        help="Candidats présentés au LLM (article : 5)")
    parser.add_argument("--embed_model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Chargement du KG ─────────────────────────────────────────────────
    with open(args.kg, encoding="utf-8") as f:
        kg_raw = json.load(f)
    kg = kg_raw if isinstance(kg_raw, list) else list(kg_raw.values())
    log.info("Triplets chargés : %d", len(kg))

    # ── Filtrage défensif : ne garder que les vraiment non mappés ─────────
    non_mapped = [
        t for t in kg
        if not t.get("canonicalized", False)
        and not t.get("predicate_definition", "").lower().startswith("(already canonical")
    ]
    log.info("Triplets non mappés (à traiter) : %d", len(non_mapped))
    if len(non_mapped) < len(kg):
        log.info("  (%d ignorés car déjà canoniques)", len(kg) - len(non_mapped))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client      = make_client(args.api_key)
    embed_model = SentenceTransformer(args.embed_model)

    results, schema = run_self_canonicalization(
        kg=non_mapped,
        client=client,
        embed_model=embed_model,
        output_path=output_path,
        top_k=args.top_k,
    )

    # ── Sauvegarde résultats ──────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Résultats → %s", output_path)

    # ── Sauvegarde schéma dynamique construit ────────────────────────────
    schema_path = output_path.parent / f"{output_path.stem}.schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, indent=2, ensure_ascii=False)
    log.info("Schéma canonique → %s  (%d relations)", schema_path, len(schema))


if __name__ == "__main__":
    main()