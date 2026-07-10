"""
quality_gates.py --- End-to-end Part 1 quality gate runner.

Extracts surfaces from phi grids, assembles STL, and returns a GateResult
with lifecycle state, meshes, phi snapshots, and STL paths.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import traceback

from geometry_contract import (
    ALLOWED_LIFECYCLE_STATES,
    MAX_EXTRACTION_RETRIES,
)
from phi_grid import PhiGrid
from surface_extraction import (
    extract_surface,
    SurfaceExtractionError,
    RadiusViolation,
    AccessibilityFailure,
    RuleViolation,
    MeshQualityFailure,
)
from stl_assembler import assemble_stl


@dataclass(frozen=True)
class GateResult:
    """Result of running quality gates on a candidate."""
    lifecycle_state: str
    meshes: Optional[dict]   # dict[str, trimesh.Trimesh] or None if failed
    phi_snapshot_paths: dict[str, str]
    stl_path: Optional[str]
    stl_half_path: Optional[str]
    failure_reason: Optional[str]

    def __post_init__(self):
        if self.lifecycle_state not in ALLOWED_LIFECYCLE_STATES:
            raise ValueError(
                f"lifecycle_state='{self.lifecycle_state}' not in allowed set "
                f"{sorted(ALLOWED_LIFECYCLE_STATES)}. This is a Part 1 bug."
            )
        if not self.phi_snapshot_paths:
            raise ValueError("phi_snapshot_paths must always be populated, even on failure.")


def run_quality_gates(
    phi_grids: dict[str, PhiGrid],
    candidate_id: str,
    out_dir: str,
    max_retries: int = MAX_EXTRACTION_RETRIES,
) -> GateResult:
    """
    Run quality gates on all phi grids.

    STEP 0: Save phi snapshots first (always).
    STEP 1-N: Retry loop for extraction and assembly.
    """
    # STEP 0: Save phi snapshots
    phi_paths = {}
    for name, phi in phi_grids.items():
        try:
            phi_paths[name] = phi.save(candidate_id, out_dir)
        except Exception as e:
            # If save fails, record an empty path so __post_init__ doesn't complain
            phi_paths[name] = ""

    # Retry loop
    for attempt in range(max_retries):
        try:
            meshes = {}
            for name, phi in phi_grids.items():
                meshes[name] = extract_surface(phi, max_radius_retries=max_retries - attempt)

            # Assemble STL
            full_path, half_path = assemble_stl(meshes, candidate_id, out_dir)

            state = "valid_simulated" if attempt == 0 else "geometry_repaired"
            return GateResult(
                lifecycle_state=state,
                meshes=meshes,
                phi_snapshot_paths=phi_paths,
                stl_path=full_path,
                stl_half_path=half_path,
                failure_reason=None,
            )

        except RadiusViolation:
            if attempt < max_retries - 1:
                continue
            return GateResult(
                lifecycle_state="geometry_rejected",
                meshes=None,
                phi_snapshot_paths=phi_paths,
                stl_path=None,
                stl_half_path=None,
                failure_reason=f"RadiusViolation after {max_retries} retries",
            )

        except AccessibilityFailure as e:
            if not e.is_large and attempt < max_retries - 1:
                continue
            state = "machining_rejected" if e.is_large else "geometry_rejected"
            return GateResult(
                lifecycle_state=state,
                meshes=None,
                phi_snapshot_paths=phi_paths,
                stl_path=None,
                stl_half_path=None,
                failure_reason=str(e),
            )

        except RuleViolation as e:
            if not e.is_major and attempt < max_retries - 1:
                continue
            return GateResult(
                lifecycle_state="rule_rejected",
                meshes=None,
                phi_snapshot_paths=phi_paths,
                stl_path=None,
                stl_half_path=None,
                failure_reason=str(e),
            )

        except MeshQualityFailure:
            if attempt < max_retries - 1:
                continue
            return GateResult(
                lifecycle_state="geometry_rejected",
                meshes=None,
                phi_snapshot_paths=phi_paths,
                stl_path=None,
                stl_half_path=None,
                failure_reason=traceback.format_exc(),
            )

        except SurfaceExtractionError:
            return GateResult(
                lifecycle_state="geometry_rejected",
                meshes=None,
                phi_snapshot_paths=phi_paths,
                stl_path=None,
                stl_half_path=None,
                failure_reason=traceback.format_exc(),
            )

    # Should not reach here, but safety net
    return GateResult(
        lifecycle_state="geometry_rejected",
        meshes=None,
        phi_snapshot_paths=phi_paths,
        stl_path=None,
        stl_half_path=None,
        failure_reason="Exhausted all retries without success or specific error.",
    )