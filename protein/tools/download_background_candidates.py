#!/usr/bin/env python3
"""Download a reproducible catalogue of structure-linked background candidates.

The script uses the public RCSB Search and Data APIs.  It writes a small TSV
catalogue and optional mmCIF files; it does not assign labels or build ratio
cohorts.  For an offline rebuild, pass ``--ids-file`` or ``--query-json`` and
keep the resulting manifest with the downloaded catalogue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_URL = "https://data.rcsb.org/rest/v1/core"
COORD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def get_json(url: str, *, payload: dict[str, Any] | None = None, timeout: int = 45) -> Any:
    data = None
    headers = {"User-Agent": "APRnet-background-catalogue/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(url: str, timeout: int = 60) -> bytes:
    request = Request(url, headers={"User-Agent": "APRnet-background-catalogue/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def canonical_sequence(value: str | None) -> str:
    if not value:
        return ""
    # RCSB canonical sequences occasionally contain line breaks or spaces.
    value = re.sub(r"\s+", "", value).upper()
    return value if value and set(value) <= AA else ""


def pdb_ids_from_search(query: dict[str, Any], max_entries: int) -> list[str]:
    start = 0
    ids: list[str] = []
    while len(ids) < max_entries:
        request = json.loads(json.dumps(query))
        options = request.setdefault("request_options", {})
        options.setdefault("paginate", {})["start"] = start
        options["paginate"]["rows"] = min(500, max_entries - len(ids))
        response = get_json(SEARCH_URL, payload=request)
        batch = [str(item.get("identifier", "")).split("_")[0].upper()
                 for item in response.get("result_set", [])]
        ids.extend(pdb for pdb in batch if re.fullmatch(r"[0-9A-Z]{4}", pdb))
        if not batch or len(batch) < options["paginate"]["rows"]:
            break
        start += len(batch)
    return list(dict.fromkeys(ids))[:max_entries]


def default_query() -> dict[str, Any]:
    # This deliberately requests experimental entries only.  Users can supply
    # a project-specific query JSON when a narrower source snapshot is needed.
    return {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.experimental_method",
                "operator": "exists",
            },
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 500}},
    }


def entity_rows(pdb_id: str) -> Iterable[dict[str, str]]:
    entry = get_json(f"{DATA_URL}/entry/{pdb_id}")
    identifiers = entry.get("rcsb_entry_container_identifiers", {})
    entity_ids = identifiers.get("polymer_entity_ids", [])
    method = ";".join(entry.get("rcsb_entry_info", {}).get("experimental_method", []) or [])
    title = entry.get("struct", {}).get("title", "")
    for entity_id in entity_ids:
        entity = get_json(f"{DATA_URL}/polymer_entity/{pdb_id}/{entity_id}")
        sequence = canonical_sequence(entity.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can"))
        if len(sequence) < 50:
            continue
        refs = entity.get("rcsb_polymer_entity_container_identifiers", {}).get(
            "reference_sequence_identifiers", [])
        uniprot = next((str(ref.get("database_accession", "")) for ref in refs
                        if ref.get("database_name", "").lower() == "uniprot"), "")
        if not uniprot:
            continue
        organisms = entity.get("rcsb_entity_source_organism", []) or []
        tax_ids = sorted({str(o.get("ncbi_taxonomy_id")) for o in organisms
                          if o.get("ncbi_taxonomy_id") is not None})
        organism_names = sorted({str(o.get("scientific_name", "")) for o in organisms if o.get("scientific_name")})
        asym_ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("asym_ids", [])
        yield {
            "pdb_id": pdb_id,
            "entity_id": str(entity_id),
            "uniprot_id": uniprot,
            "sequence": sequence,
            "sequence_length": str(len(sequence)),
            "chain_ids": ";".join(map(str, asym_ids)),
            "organism_id": ";".join(tax_ids),
            "organism": ";".join(organism_names),
            "species_group": "human" if "9606" in tax_ids else "other",
            "scope": "deposited_construct",
            "protein_name": entity.get("rcsb_polymer_entity", {}).get("pdbx_description", ""),
            "experimental_methods": method,
            "entry_title": title,
            "structure_url": f"https://www.rcsb.org/structure/{pdb_id}",
            "source_url": f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}",
            "coordinate_url": COORD_URL.format(pdb_id=pdb_id),
        }


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output catalogue directory")
    parser.add_argument("--ids-file", type=Path, help="Text file with one RCSB PDB ID per line")
    parser.add_argument("--query-json", type=Path, help="RCSB Search API request JSON")
    parser.add_argument("--max-entries", type=int, default=500,
                        help="Maximum entries to inspect (default: 500)")
    parser.add_argument("--download-structures", action="store_true",
                        help="Download one mmCIF per selected PDB entry")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Delay between API requests in seconds")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.ids_file:
        pdb_ids = list(dict.fromkeys(re.findall(r"[0-9A-Z]{4}", args.ids_file.read_text().upper())))
        pdb_ids = pdb_ids[:args.max_entries]
        request = {"mode": "ids-file", "path": str(args.ids_file)}
    else:
        query = json.loads(args.query_json.read_text()) if args.query_json else default_query()
        request = query
        pdb_ids = pdb_ids_from_search(query, args.max_entries)
    rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for index, pdb_id in enumerate(pdb_ids, start=1):
        try:
            rows.extend(entity_rows(pdb_id))
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as exc:
            errors.append({"pdb_id": pdb_id, "error": f"{type(exc).__name__}: {exc}"})
        if args.sleep:
            time.sleep(args.sleep)
        if index % 50 == 0:
            print(f"inspected {index}/{len(pdb_ids)} entries", file=sys.stderr)
    # One entity per UniProt/PDB pair is the unit consumed by the cohort builder.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        unique.setdefault((row["pdb_id"], row["uniprot_id"]), row)
    rows = sorted(unique.values(), key=lambda row: (row["pdb_id"], row["uniprot_id"]))
    if args.download_structures:
        structure_dir = args.output / "structures"
        structure_dir.mkdir(exist_ok=True)
        for pdb_id in sorted({row["pdb_id"] for row in rows}):
            target = structure_dir / f"{pdb_id}.cif"
            try:
                if not target.exists():
                    target.write_bytes(get_bytes(COORD_URL.format(pdb_id=pdb_id)))
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                for row in rows:
                    if row["pdb_id"] == pdb_id:
                        row["structure_file"] = target.name
                        row["structure_path"] = str(target)
                        row["structure_sha256"] = digest
            except (HTTPError, URLError, TimeoutError) as exc:
                errors.append({"pdb_id": pdb_id, "error": f"coordinate: {type(exc).__name__}: {exc}"})
    write_tsv(args.output / "candidates.tsv", rows)
    if errors:
        write_tsv(args.output / "errors.tsv", errors)
    manifest = {
        "schema": "APRnet.background_candidates.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api": {"search": SEARCH_URL, "data": DATA_URL, "coordinates": "RCSB files API"},
        "request": request,
        "ids_requested": len(pdb_ids),
        "entities_written": len(rows),
        "errors": len(errors),
        "downloaded_structures": bool(args.download_structures),
        "candidate_sha256": hashlib.sha256((args.output / "candidates.tsv").read_bytes()).hexdigest(),
        "selection_warning": "Candidates are not labels; cohort construction assigns source-screened assumed background negatives.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
