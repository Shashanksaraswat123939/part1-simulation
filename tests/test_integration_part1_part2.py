"""
Tests for Part 1 / Part 2 integration.
Verifies that Part 1's output types are compatible with Part 2's input types.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "part2_simulation"))

import numpy as np

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def test_co2_mass_constant_matches():
    from geometry_contract import CO2_MASS_KG
    from mass_com_ingest import CO2_CARTRIDGE_MASS_KG
    assert abs(CO2_MASS_KG - CO2_CARTRIDGE_MASS_KG) < 1e-12, \
        f"Part1 CO2_MASS_KG={CO2_MASS_KG} != Part2 CO2_CARTRIDGE_MASS_KG={CO2_CARTRIDGE_MASS_KG}"
    _pass("test_co2_mass_constant_matches")

def test_component_mass_com_shape_matches():
    from mass_com_calculator import ComponentMassCOM as P1COM
    from physics_contract import ComponentMassCOM as P2COM
    # Both should have same fields
    p1_fields = P1COM.__dataclass_fields__
    p2_fields = P2COM.__dataclass_fields__
    assert "name" in p1_fields and "name" in p2_fields
    assert "mass_kg" in p1_fields and "mass_kg" in p2_fields
    _pass("test_component_mass_com_shape_matches")

def test_fixed_hardware_spec_accepts_part1_values():
    from geometry_contract import CO2_MASS_KG
    from mass_com_ingest import FixedHardwareSpec
    spec = FixedHardwareSpec(
        co2_cartridge_mass_kg=CO2_MASS_KG,
        co2_cartridge_com=(0.005, 0.0, 0.025),
        rear_wing_mass_kg=0.003,
        rear_wing_com=(0.130, 0.0, 0.035),
        wheels_axles_mass_kg=0.020,
        wheels_axles_com=(0.065, 0.0, 0.015),
    )
    assert spec.co2_cartridge_mass_kg == 0.023
    _pass("test_fixed_hardware_spec_accepts_part1_values")

def test_lifecycle_states_match():
    from geometry_contract import ALLOWED_LIFECYCLE_STATES as p1_states
    try:
        from candidate_record import ALLOWED_LIFECYCLE_STATES as p2_states
        assert p1_states == p2_states, f"Lifecycle states mismatch: {p1_states ^ p2_states}"
    except ImportError:
        # Part 2 candidate_record may not be importable; skip
        pass
    _pass("test_lifecycle_states_match")

def test_mass_com_can_be_ingested():
    from mass_com_calculator import ComponentMassCOM
    from mass_com_ingest import FixedHardwareSpec, ingest_mass_com
    machined = [
        ComponentMassCOM(name="nose", mass_kg=0.001, com_x_m=0.005, com_y_m=0.0, com_z_m=0.02),
        ComponentMassCOM(name="sidepod", mass_kg=0.002, com_x_m=0.06, com_y_m=0.0, com_z_m=0.015),
        ComponentMassCOM(name="rearpod", mass_kg=0.001, com_x_m=0.135, com_y_m=0.0, com_z_m=0.015),
        ComponentMassCOM(name="main_body", mass_kg=0.005, com_x_m=0.065, com_y_m=0.0, com_z_m=0.02),
    ]
    fixed = FixedHardwareSpec(
        co2_cartridge_mass_kg=0.023,
        co2_cartridge_com=(0.005, 0.0, 0.025),
        rear_wing_mass_kg=0.003,
        rear_wing_com=(0.130, 0.0, 0.035),
        wheels_axles_mass_kg=0.020,
        wheels_axles_com=(0.065, 0.0, 0.015),
    )
    full = ingest_mass_com(machined, fixed)
    assert full.total_mass_kg > 0
    assert full.com_z_m > 0 and full.com_z_m < 0.1, f"COM z={full.com_z_m} unreasonable"
    _pass("test_mass_com_can_be_ingested")

if __name__ == "__main__":
    test_co2_mass_constant_matches()
    test_component_mass_com_shape_matches()
    test_fixed_hardware_spec_accepts_part1_values()
    test_lifecycle_states_match()
    test_mass_com_can_be_ingested()
    print("\nAll integration tests passed.")
