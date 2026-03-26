#!/usr/bin/env python3
"""
generate_obscore.py
-------------------
Generate ObsCore-compliant CSV lines for MAGIC ECSV data products (SEDs & light curves)
so they can be ingested into a TAP service.

Each ECSV file listed under "File list MAGIC:" in the YAML manifest produces
exactly one ObsCore row.

Usage
-----
    python generate_obscore.py magic_2025h.yaml [options]

Options
-------
    -o, --output   FILE   Output CSV path (default: <yaml_stem>_obscore.csv)
    -d, --ecsv-dir DIR    Directory containing ECSV files (default: YAML directory)
    -u, --base-url URL    Override Flink from YAML for building access_url
    --no-header           Omit the CSV header row

Requirements
------------
    pip install astropy numpy

Notes
-----
    * The YAML format used by MAGIC mixes "key = value" (File_info section) and
      "key: value" (Paper info / Sources / File lists) styles; a dedicated custom
      parser handles both without requiring strict YAML compliance.
    * Filenames containing '_sed' → dataproduct_type = "sed"
      Filenames containing '_lc'  → dataproduct_type = "light-curve"
    * SED numbering and LC numbering are independent counters (first SED → /1,
      first LC → /1, second SED → /2, …) as reflected in obs_publisher_id.
    * texpo in ECSV is expected in hours (unit: h); it is converted to seconds
      for the t_exptime column.  For SEDs the first row value is used (all rows
      share the same observation window).  For light curves the first row is
      also used by default; adjust exptime_seconds() if summing bins is needed.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.table import Table


# ── Fixed / always-constant ObsCore values ────────────────────────────────────

DEFAULTS = dict(
    calib_level     = 4,
    obs_collection  = "MAGIC/DL5",
    access_format   = "ecsv",
    access_estsize  = 10,
    facility_name   = "MAGIC",
    instrument_name = "MAGIC-stereo",
    s_fov           = 3.5,
    s_region        = 3.1,
    s_resolution    = 360,
    s_xel1          = -1,
    s_xel2          = -1,
    t_resolution    = -1,
    t_xel           = -1,
    em_min          = "2E-21",
    em_max          = "2E-16",
    em_res_power    = 10,
    em_xel          = -1,
    o_ucd           = "phot.flux.density",
    pol_xel         = -1,
    target_class    = "NULL",
    data_rights     = "Public",
)

OBSCORE_COLUMNS = [
    "dataproduct_type", "calib_level", "obs_collection", "obs_id",
    "obs_publisher_id", "access_url", "access_format", "access_estsize",
    "facility_name", "instrument_name", "target_name", "s_ra", "s_dec",
    "s_fov", "s_region", "s_resolution", "s_xel1", "s_xel2",
    "t_min", "t_max", "t_exptime", "t_resolution", "t_xel",
    "em_min", "em_max", "em_res_power", "em_xel", "o_ucd",
    "pol_xel", "target_class", "obs_creation_date", "obs_creator_name",
    "bib_reference", "data_rights",
]


# ── Custom YAML-like parser ────────────────────────────────────────────────────

def parse_magic_yaml(path: Path) -> dict:
    """
    Parse the non-standard YAML-like metadata file used by MAGIC.

    Grammar handled:
      Section header   : "Section name:"   (no leading whitespace, ends with colon)
      key = value      : "  Fdate = 2026.03.18"
      key: value       : "  Ptitle: 'Some title'"
      bare list entry  : "  magic_2025h_fig2_sed.ecsv"  (no = or :  →  list item)

    Returns a nested dict where each top-level key is a section name and its
    value is either a dict of key→value pairs or a list of bare strings.
    """
    result: dict = {}
    current_section: Optional[str] = None

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip()
            stripped = line.strip()

            # Skip blank lines
            if not stripped:
                continue

            # ── Section header (no leading space, contains ':') ────────────
            if raw and not raw[0].isspace():
                colon_idx = stripped.find(':')
                if colon_idx >= 0:
                    section_name = stripped[:colon_idx].strip()
                    inline_val   = stripped[colon_idx + 1:].strip().strip('"').strip("'")
                    current_section = section_name
                    # Start as dict; if inline value present keep it as string temporarily
                    result[current_section] = inline_val if inline_val else {}
                continue

            # ── Content within a section ───────────────────────────────────
            if current_section is None:
                continue

            # Promote string (from inline section header value) to dict
            if isinstance(result[current_section], str):
                result[current_section] = {}

            has_eq    = '=' in stripped
            colon_pos = stripped.find(':')
            has_colon = colon_pos >= 0

            if has_eq:
                # key = value  (used in File_info section)
                key, _, value = stripped.partition('=')
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if isinstance(result[current_section], list):
                    result[current_section] = {}
                result[current_section][key] = value

            elif has_colon:
                # key: value  (used in Paper info, Sources, …)
                key   = stripped[:colon_pos].strip()
                value = stripped[colon_pos + 1:].strip().strip('"').strip("'")
                if isinstance(result[current_section], list):
                    result[current_section] = {}
                result[current_section][key] = value

            else:
                # Bare value → append to list  (used for file names)
                if not isinstance(result[current_section], list):
                    result[current_section] = []
                result[current_section].append(stripped)

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def dataproduct_type(filename: str) -> str:
    """
    Infer ObsCore dataproduct_type from the ECSV filename.

    Rules (applied in order, case-insensitive):
      * filename contains '_sed'  → 'sed'
      * filename contains '_lc'   → 'timeseries'

    A ValueError is raised when neither token is found so the caller can
    skip the file with a clear warning rather than silently defaulting.
    """
    stem = Path(filename).stem.lower()
    # Walk the underscore-separated tokens so '_lc' and '_sed' are never
    # confused with substrings of longer words (e.g. '_lc' inside '_lca').
    tokens = stem.split("_")
    if "sed" in tokens:
        return "sed"
    if "lc" in tokens:
        return "light-curve"
    # Fallback: scan as substring (handles _SED., _LC. without surrounding _)
    if "_sed" in stem:
        return "sed"
    if "_lc" in stem:
        return "light-curve"
    raise ValueError(
        f"Cannot determine dataproduct_type for '{filename}': "
        "filename must contain '_sed' or '_lc'."
    )


def dataset_code(yaml_stem: str) -> str:
    """
    Convert the YAML stem into the VO dataset path component.

    Examples:
        'magic_2025h'  →  'magic/2025h'
        'magic_2024ab' →  'magic/2024ab'
    """
    m = re.match(r'^(magic)_(.+)$', yaml_stem, re.IGNORECASE)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # Fallback: replace first underscore
    return yaml_stem.replace("_", "/", 1)


def get_sources_section(meta: dict) -> dict:
    """
    Return the sources/targets dict regardless of the section header used.

    Handles both 'Sources in file' and 'Targets in file' (and any other
    capitalisation variant containing 'source' or 'target').
    """
    for key, val in meta.items():
        if re.search(r"source|target", key, re.IGNORECASE) and isinstance(val, dict):
            return val
    return {}


def all_source_names(sources: dict) -> list[tuple[str, str, str]]:
    """
    Return a list of (name, ra, dec) triples from the sources section,
    in index order (Tpname01, Tpname02, …).
    """
    result = []
    idx = 1
    while True:
        name_key = f"Tpname{idx:02d}"
        if name_key not in sources:
            break
        ra  = str(sources.get(f"Tra{idx:02d}",  "")).strip()
        dec = str(sources.get(f"Tdec{idx:02d}", "")).strip()
        result.append((str(sources[name_key]).strip(), ra, dec))
        idx += 1
    return result


def find_source_coords(meta: dict, filename: str):
    """
    Return (target_name, ra, dec) for the source associated with *filename*.

    Strategy
    --------
    1. Build a list of all (name, ra, dec) triples from the YAML sources
       section (handles both 'Sources in file' and 'Targets in file').
    2. Try to match each source name against the filename stem
       (whitespace-stripped, case-insensitive).
    3. If exactly one match is found, use it.
    4. If no match is found (common for single-source YAMLs whose file names
       do not contain the source name), fall back to the first entry (Tpname01 /
       Tra01 / Tdec01).
    5. If the sources section is empty altogether, return ('Unknown', None, None).
    """
    sources = get_sources_section(meta)
    entries = all_source_names(sources)

    if not entries:
        return "Unknown", None, None

    stem_clean = re.sub(r"\s+", "", Path(filename).stem).lower()

    matches = [
        (name, ra, dec)
        for name, ra, dec in entries
        if re.sub(r"\s+", "", name).lower() in stem_clean
    ]

    if len(matches) == 1:
        name, ra, dec = matches[0]
    elif len(matches) > 1:
        # Multiple names found in the stem — pick the longest (most specific)
        matches.sort(key=lambda t: len(t[0]), reverse=True)
        name, ra, dec = matches[0]
    else:
        # No name found in filename → fall back to first source in YAML
        name, ra, dec = entries[0]

    return name, ra or None, dec or None


def exptime_seconds(table: Table) -> float:
    """
    Return the exposure time in seconds from the ECSV 'texpo' column.

    The ECSV standard unit for this column is hours ('h'); the value is
    converted to seconds automatically.  For SED files all rows share the
    same observation window, so the first non-NaN value is used.
    """
    if "texpo" not in table.colnames:
        return -1.0

    col = table["texpo"]
    unit = str(col.unit).lower() if hasattr(col, "unit") else ""
    vals = np.array(col, dtype=float)
    valid = vals[~np.isnan(vals)]

    if len(valid) == 0:
        return -1.0

    expo = float(valid[0])
    if "h" in unit:
        expo *= 3600.0
    return expo


# ── Core row builder ──────────────────────────────────────────────────────────

def make_obscore_row(
    ecsv_path: Path,
    meta: dict,
    yaml_stem: str,
    file_index: int,
    ftype: str,
    base_url: Optional[str],
) -> dict:
    """
    Build one ObsCore dict from an ECSV file and the shared YAML metadata.

    Parameters
    ----------
    ecsv_path  : path to the ECSV file
    meta       : parsed YAML dict
    yaml_stem  : YAML filename without extension  (e.g. 'magic_2025h')
    file_index : 1-based counter within this dataproduct_type
    ftype      : 'sed' or 'timeseries'
    base_url   : optional override for the Flink base URL
    """
    table = Table.read(ecsv_path, format="ascii.ecsv")

    # ── Target name & sky coordinates ─────────────────────────────────────────
    # Coordinates always come from the YAML (Tra/Tdec), matched by filename.
    # The target_name from the YAML is authoritative; the srcname column in the
    # ECSV may use an instrument-specific alias.
    target_name, s_ra, s_dec = find_source_coords(meta, ecsv_path.name)

    # ── Time range (MJD) ──────────────────────────────────────────────────────
    # Light curves: span from the lowest to the highest individual time stamp.
    # SEDs: span from the earliest tstart to the latest tstop.
    if ftype == "light-curve" and "t" in table.colnames:
        t_vals = np.array(table["t"], dtype=float)
        t_min  = float(np.nanmin(t_vals))
        t_max  = float(np.nanmax(t_vals))
    elif "tstart" in table.colnames and "tstop" in table.colnames:
        t_min = float(np.nanmin(np.array(table["tstart"], dtype=float)))
        t_max = float(np.nanmax(np.array(table["tstop"],  dtype=float)))
    else:
        t_min = t_max = -1.0

    t_exptime = exptime_seconds(table)

    # ── Bibliographic / provenance metadata ───────────────────────────────────
    paper = meta.get("Paper info", {}) or {}
    finfo = meta.get("File_info",  {}) or {}

    doi               = str(paper.get("Pdoi",  "")).strip()
    obs_creation_date = str(finfo.get("Fdate", "")).strip()

    # obs_creator_name: use the generator e-mail address from Fgen
    obs_creator_name = str(finfo.get("Fgen", "")).strip()

    # ── Identifiers & access URL ──────────────────────────────────────────────
    dset = dataset_code(yaml_stem)   # e.g. "magic/2025h"

    obs_publisher_id = (
        f"ivo://magictelescope/dataset?{dset}/{ftype}/{file_index}"
    )

    # Build access URL:  <Flink>/ecsv/<year_code>_<SourceNoSpaces>/<filename>
    flink = str(finfo.get("Flink", "")).strip().strip("<>").strip()
    if base_url:
        flink = base_url.rstrip("/")

    year_code = dset.split("/")[-1]               # e.g. "2025h"
    src_clean = re.sub(r"\s+", "", target_name)  # e.g. "NGC4151", "130701A"

    if flink:
        access_url = f"{flink}/ecsv/{year_code}_{src_clean}/{ecsv_path.name}"
    else:
        # Fallback: just the filename (user should set --base-url)
        access_url = ecsv_path.name

    # ── Assemble final row ────────────────────────────────────────────────────
    row = {**DEFAULTS}   # start from fixed defaults
    row.update(
        dataproduct_type  = ftype,
        obs_id            = yaml_stem,
        obs_publisher_id  = obs_publisher_id,
        access_url        = access_url,
        target_name       = target_name,
        s_ra              = s_ra,
        s_dec             = s_dec,
        t_min             = t_min,
        t_max             = t_max,
        t_exptime         = int(t_exptime) if t_exptime >= 0 else -1,
        obs_creation_date = obs_creation_date,
        obs_creator_name  = obs_creator_name,
        bib_reference     = doi,
    )
    return row


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate ObsCore CSV lines for MAGIC ECSV data products.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("yaml_file",
                    help="YAML manifest file (e.g. magic_2025h.yaml)")
    ap.add_argument("-o", "--output",   metavar="FILE",
                    help="Output CSV file (default: <yaml_stem>_obscore.csv)")
    ap.add_argument("-d", "--ecsv-dir", metavar="DIR",
                    help="Directory containing ECSV files (default: YAML directory)")
    ap.add_argument("-u", "--base-url", metavar="URL",
                    help="Base URL to override Flink for access_url construction")
    ap.add_argument("--no-header", action="store_true",
                    help="Omit the CSV header row")
    args = ap.parse_args()

    yaml_path = Path(args.yaml_file).resolve()
    if not yaml_path.exists():
        sys.exit(f"ERROR: YAML file not found: '{yaml_path}'")

    yaml_stem = yaml_path.stem                          # e.g. "magic_2025h"
    ecsv_dir  = Path(args.ecsv_dir).resolve() if args.ecsv_dir else yaml_path.parent
    out_path  = (
        Path(args.output)
        if args.output
        else yaml_path.parent / f"{yaml_stem}_obscore.csv"
    )

    # ── Parse YAML ────────────────────────────────────────────────────────────
    print(f"Parsing YAML: {yaml_path}")
    try:
        meta = parse_magic_yaml(yaml_path)
    except Exception as exc:
        sys.exit(f"ERROR: cannot parse '{yaml_path}': {exc}")

    # ── Collect ECSV file list ────────────────────────────────────────────────
    file_list = meta.get("File list MAGIC", [])
    if isinstance(file_list, dict):
        file_list = [v for v in file_list.values() if v]
    if isinstance(file_list, str):
        file_list = [file_list] if file_list else []

    # Strip any stray whitespace
    file_list = [str(f).strip() for f in file_list if str(f).strip()]

    if not file_list:
        sys.exit("ERROR: no files found under 'File list MAGIC:' in the YAML.")

    # Separate by type, preserving original order within each type
    sed_files, lc_files = [], []
    for f in file_list:
        try:
            ftype = dataproduct_type(f)
        except ValueError as exc:
            print(f"  WARNING: {exc} – skipping.", file=sys.stderr)
            continue
        (sed_files if ftype == "sed" else lc_files).append(f)

    print(f"Found {len(sed_files)} SED file(s) and {len(lc_files)} LC file(s).\n")

    # ── Process files ─────────────────────────────────────────────────────────
    rows = []

    for collection, ftype in ((sed_files, "sed"), (lc_files, "light-curve")):
        for idx, fname in enumerate(collection, start=1):
            ecsv_path = ecsv_dir / fname
            if not ecsv_path.exists():
                print(f"  WARNING: '{fname}' not found in '{ecsv_dir}' – skipping.",
                      file=sys.stderr)
                continue
            try:
                row = make_obscore_row(
                    ecsv_path, meta, yaml_stem, idx, ftype, args.base_url
                )
                rows.append(row)
                print(f"  [{ftype.upper():10s}]  #{idx:02d}  {fname}"
                      f"  →  target={row['target_name']}"
                      f"  t=[{row['t_min']}, {row['t_max']}]")
            except Exception as exc:
                print(f"  WARNING: error processing '{fname}': {exc}",
                      file=sys.stderr)

    if not rows:
        sys.exit("ERROR: no ObsCore rows were generated.")

    # ── Write output CSV ──────────────────────────────────────────────────────
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBSCORE_COLUMNS)
        if not args.no_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓  {len(rows)} row(s) written to '{out_path}'")


if __name__ == "__main__":
    main()
