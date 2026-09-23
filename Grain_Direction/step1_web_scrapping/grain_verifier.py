def verify_grain_expected(material):
    """
    Analyzes the material description to determine if the product SHOULD
    have a directional grain.

    Returns "YES", "NO", or "UNKNOWN".

    Logic:
      - Checks name + finish + description for grain/no-grain keywords.
      - More specific keywords are checked first.
      - If truly ambiguous, returns "UNKNOWN" (which triggers visual analysis
        when an image IS available, or defaults to "NO" as a safe fallback
        when no image is available).
    """
    desc = " ".join([
        material.get('name', ''),
        material.get('code_or_finish', ''),
        material.get('description', ''),
        material.get('manufacturer', ''),
    ]).lower()

    # ── SPECIFIC OVERRIDES (brand+finish combos that override general keywords) ──
    # Wilsonart traceless is an ultra-matte solid finish (no grain)
    specific_no_patterns = [
        ("wilsonart", "traceless"),
        ("formica", "solid surface"),
        ("formica", "everform"),
    ]
    for brand, finish_kw in specific_no_patterns:
        if brand in desc and finish_kw in desc:
            return "NO"

    # ── YES-grain keywords (check before generic NO keywords) ─────────────
    grain_keywords = [
        # Explicit grain words
        "grain", "wood grain", "natural grain", "rift", "rift cut", "quarter sawn",
        "quarter-sawn", "plain sliced", "flat cut",
        # Wood types (almost always have grain)
        "wood", "veneer", "tambour", "slat",
        "ash", "cedar", "oak", "walnut", "maple", "cherry", "birch",
        "mahogany", "teak", "pine", "spruce", "poplar",
        # Generic wood products
        "plywood",
        # Wilsonart / laminate patterns known to be wood-grain
        "landmark wood", "heritage wood", "natural recon",
    ]

    # ── NO-grain keywords ─────────────────────────────────────────────────
    no_grain_keywords = [
        # Finish descriptors that mean no grain
        "softgrain", "soft grain", "traceless", "matte finish", "solid surface",
        "solid color", "solid colour", "polished finish", "matte",
        # Material types with no directional grain
        "quartz", "terrazzo", "concrete", "glass", "ceramic", "porcelain",
        "metal", "stainless steel", "stainless", "steel", "aluminum", "aluminium",
        "brass", "bronze", "copper", "iron",
        "melamine", "lacquer", "paint", "painted",
        "fabric", "upholstery", "leather", "felt",
        "resin", "acrylic",
        # Solid surface brands
        "corian", "formica everform", "formica solid surface", "wilsonart solid",
        "hi-macs", "avonite", "staron",
        # Specific laminates known to be non-grain
        "dove grey", "snow white", "bleached concrete", "alabaster terrazzo",
        "white melamine", "white cab liner",
        # Plywood/panel when primed/painted (no visible grain required)
        "primed", "delivered primed",
    ]

    # ── YES-grain keywords ────────────────────────────────────────────────

    # Check YES-grain FIRST (plywood, wood, veneer etc. win over 'primed')
    for kw in grain_keywords:
        if kw in desc:
            return "YES"

    # Then check NO-grain
    for kw in no_grain_keywords:
        if kw in desc:
            return "NO"

    return "UNKNOWN"


def description_based_grain_decision(material):
    """
    Production fallback: when web scrape fails AND visual analysis is impossible,
    use description keywords to return a final grain decision.

    Returns dict with: grain (YES/NO/UNKNOWN), direction, analysis string
    """
    result = verify_grain_expected(material)

    if result == "YES":
        return {
            "grain": "YES",
            "direction": "UNKNOWN",
            "analysis": "Grain: Yes | Direction: Unknown (Determined by description keywords — no image available)"
        }
    elif result == "NO":
        return {
            "grain": "NO",
            "direction": "NONE",
            "analysis": "Grain: No | Direction: None (Determined by description keywords — no image available)"
        }
    else:
        # Truly ambiguous — default to NO (conservative: don't flag unnecessarily)
        name = material.get('name', '')
        manufacturer = material.get('manufacturer', '')
        return {
            "grain": "NO",
            "direction": "NONE",
            "analysis": f"Grain: No | Direction: None (Defaulted — could not determine from '{manufacturer} {name}')"
        }


if __name__ == "__main__":
    # Test cases
    test_materials = [
        {"tag": "PL-C", "name": "'LANDMARK WOOD' 7981K-12", "code_or_finish": "SOFTGRAIN FINISH", "manufacturer": "WILSONART"},
        {"tag": "PL-D", "name": "'SPRUCE VELVET' 15518-31", "code_or_finish": "TRACELESS FINISH", "manufacturer": "WILSONART"},
        {"tag": "PL-F", "name": "'CRYSTAL' D386-60",        "code_or_finish": "MATTE FINISH",     "manufacturer": "WILSONART"},
        {"tag": "QZ-B", "name": "'ALABASTER TERRAZZO'",     "code_or_finish": "2cm, POLISHED FINISH", "manufacturer": "CORIAN"},
        {"tag": "SS-A", "name": "EVERFORM SOLID SURFACE 'BLEACHED CONCRETE'", "code_or_finish": "601, 1/2 THICK", "manufacturer": "FORMICA"},
        {"tag": "PL-G", "name": "PLYWOOD, DELIVERED PRIMED", "code_or_finish": "", "manufacturer": ""},
        {"tag": "QZ-D", "name": "LONDON GREY",              "code_or_finish": "",                  "manufacturer": "CAESARSTONE"},
        {"tag": "WD-A", "name": "OAK VENEER",               "code_or_finish": "NATURAL GRAIN",     "manufacturer": ""},
    ]

    print(f"{'Tag':<8} {'verify_grain_expected':<25} {'description_based_decision'}")
    print("-" * 70)
    for m in test_materials:
        vge = verify_grain_expected(m)
        dbd = description_based_grain_decision(m)
        print(f"{m['tag']:<8} {vge:<25} grain={dbd['grain']} | {dbd['analysis'][:50]}")
