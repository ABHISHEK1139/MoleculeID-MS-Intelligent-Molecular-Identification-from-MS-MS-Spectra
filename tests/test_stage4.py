"""Unit tests for Stage 4: Formula engine and Candidate Database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from src.core.formula import (
    parse_formula,
    format_formula,
    formula_monoisotopic_mass,
    calculate_rdbe,
    validate_seven_golden_rules,
    generate_candidate_formulas,
)
from src.search.candidate_generator import CandidateDatabase


def test_formula_parsing_and_formatting():
    formula = "C20H15N3O2"
    comp = parse_formula(formula)
    assert comp == {"C": 20, "H": 15, "N": 3, "O": 2}, f"Unexpected comp: {comp}"
    rebuilt = format_formula(comp)
    assert rebuilt == formula, f"Rebuilt formula {rebuilt} != {formula}"

    # Test Hill system order (C first, H second, then alphabetical)
    comp_unordered = {"O": 2, "H": 15, "C": 20, "N": 3}
    assert format_formula(comp_unordered) == "C20H15N3O2"


def test_formula_mass_accuracy():
    smiles_cases = [
        ("C20H15N3O2", "O=C1NC(=O)c2cc(Nc3ccccc3)c(Nc3ccccc3)cc21"),
        ("C8H10N4O2", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),  # Caffeine
        ("C6H12O6", "OCC1OC(O)C(O)C(O)C1O"),  # Glucose
    ]
    for expected_f, smi in smiles_cases:
        mol = Chem.MolFromSmiles(smi)
        rdkit_mw = ExactMolWt(mol)
        calc_mw = formula_monoisotopic_mass(expected_f)
        assert calc_mw is not None
        diff = abs(calc_mw - rdkit_mw)
        assert diff < 1e-4, f"Mass mismatch for {expected_f}: {calc_mw} vs RDKit {rdkit_mw} (diff={diff})"


def test_seven_golden_rules():
    # Valid metabolite
    caffeine_comp = {"C": 8, "H": 10, "N": 4, "O": 2}
    valid, reason = validate_seven_golden_rules(caffeine_comp)
    assert valid, f"Caffeine failed golden rules: {reason}"

    # Invalid: odd valence violation (e.g. C8H11N4O2 has odd valence sum)
    invalid_odd = {"C": 8, "H": 11, "N": 4, "O": 2}
    valid, reason = validate_seven_golden_rules(invalid_odd)
    assert not valid, "Failed to catch Senior's 2nd rule violation"
    assert "Senior's 2nd rule" in reason

    # Invalid: negative RDBE (e.g. C2H10)
    neg_rdbe = {"C": 2, "H": 10}
    valid, reason = validate_seven_golden_rules(neg_rdbe)
    assert not valid, "Failed to catch negative RDBE"

    # Invalid: non-physical H/C ratio (e.g. C10H45)
    high_hc = {"C": 10, "H": 46}
    valid, reason = validate_seven_golden_rules(high_hc)
    assert not valid, "Failed to catch high H/C ratio"


def test_de_novo_formula_generation():
    # Test on caffeine neutral mass: 194.080376
    caffeine_mass = 194.080376
    candidates = generate_candidate_formulas(caffeine_mass, ppm_tol=15.0)
    found_formulas = [c["formula"] for c in candidates]
    assert "C8H10N4O2" in found_formulas, f"C8H10N4O2 not found in generated: {found_formulas}"
    for c in candidates:
        assert c["ppm_error"] <= 15.0
        assert c["rdbe"] >= -0.5


def test_candidate_database_indexing_and_querying():
    test_molecules = ["mol1", "mol2", "mol3", "mol4"]
    test_smiles = {
        "mol1": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # Caffeine (194.0804)
        "mol2": "OCC1OC(O)C(O)C(O)C1O",        # Glucose (180.0634)
        "mol3": "CC(=O)Oc1ccccc1C(=O)O",       # Aspirin (180.0423)
        "mol4": "O=C1NC(=O)c2cc(Nc3ccccc3)c(Nc3ccccc3)cc21",  # Large (329.1164)
    }
    db = CandidateDatabase(test_molecules, test_smiles)
    assert len(db) == 4

    # Query near caffeine with [M+H]+: precursor ~195.0877
    matches = db.query_two_tier(195.0877, adduct="[M+H]+", ppm_primary=20.0)
    assert len(matches) >= 1
    assert matches[0].mol == "mol1"
    assert matches[0].tier == 1
    assert matches[0].weight == 1.0

    # Query near Glucose (180.0634) vs Aspirin (180.0423)
    # Glucose [M+H]+ ~181.0712
    matches_glu = db.query_two_tier(181.0712, adduct="[M+H]+", ppm_primary=20.0, ppm_fallback=150.0)
    # Glucose should be Tier 1 (within 20 ppm)
    tier1_mols = [m.mol for m in matches_glu if m.tier == 1]
    assert "mol2" in tier1_mols
    # Aspirin difference is ~117 ppm, so in fallback (Tier 2)
    tier2_mols = [m.mol for m in matches_glu if m.tier == 2]
    assert "mol3" in tier2_mols


if __name__ == "__main__":
    print("[test] running test_formula_parsing_and_formatting...")
    test_formula_parsing_and_formatting()
    print("[test] running test_formula_mass_accuracy...")
    test_formula_mass_accuracy()
    print("[test] running test_seven_golden_rules...")
    test_seven_golden_rules()
    print("[test] running test_de_novo_formula_generation...")
    test_de_novo_formula_generation()
    print("[test] running test_candidate_database_indexing_and_querying...")
    test_candidate_database_indexing_and_querying()
    print("\nALL STAGE 4 UNIT TESTS PASSED!")
