import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.validate_sign_separated_data_roles import (
    ROLE_PLAN_PATH,
    run_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SignSeparatedDataRolePreflightTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        plan = json.loads((ROOT / ROLE_PLAN_PATH).read_text(encoding="utf-8"))

        copied_paths = {
            role["manifest"]["path"]
            for role in plan["roles"].values()
            if isinstance(role.get("manifest"), dict)
        }
        copied_paths.add(plan["baseline"]["current_second_stage"]["inference_script"]["path"])
        copied_paths.add("scripts/train/train_patch_cleanup_erasemap_probe.py")
        for relative in copied_paths:
            source = ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

        primary = root / "artifacts/current-primary"
        primary.mkdir(parents=True)
        primary_config = primary / "config.yaml"
        primary_checkpoint = primary / "micro_region_probe_step0001.pth"
        primary_config.write_text("model: fixture\n", encoding="utf-8")
        primary_checkpoint.write_bytes(b"primary-checkpoint")
        second_stage = root / "artifacts/current-second-stage-best.pt"
        second_stage.write_bytes(b"second-stage-checkpoint")

        plan["baseline"]["current_primary"]["config"]["sha256"] = sha256_file(
            primary_config
        )
        plan["baseline"]["current_primary"]["checkpoint"][
            "sha256"
        ] = sha256_file(primary_checkpoint)
        plan["baseline"]["current_second_stage"]["checkpoint"][
            "sha256"
        ] = sha256_file(second_stage)
        role_plan_path = root / ROLE_PLAN_PATH
        role_plan_path.parent.mkdir(parents=True, exist_ok=True)
        role_plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        audit_path = (
            root
            / "outputs/sign-separated-residual-repair-synthetic-preflight-20260809"
            / "audit-final.json"
        )
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(
            json.dumps(
                {
                    "terminal": "PASS",
                    "model_type": "sign_separated_residual_delta",
                    "exact_identity_init": True,
                    "zero_magnitude_projection_init": True,
                    "has_global_scale": False,
                    "training_cli_enabled": False,
                    "opposed_pixel_count": 0,
                    "serialization_exact": True,
                    "residual_delta_bound": 0.08,
                    "gradient_cases": [
                        {
                            "direction": 1,
                            "bright_gradient_abs": 0.08,
                            "dark_gradient_abs": 0.0,
                            "route_gradient_abs": 1.0,
                        },
                        {
                            "direction": -1,
                            "bright_gradient_abs": 0.0,
                            "dark_gradient_abs": 0.08,
                            "route_gradient_abs": 1.0,
                        },
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        ledger = {
            "baseline": {
                "config": plan["baseline"]["current_primary"]["config"],
                "checkpoint": plan["baseline"]["current_primary"]["checkpoint"],
            },
            "active_iteration": {
                "id": "sign-separated-residual-repair",
                "prerequisites": [
                    {
                        "id": "sign_separated_residual_synthetic_preflight",
                        "status": "passed",
                    },
                    {
                        "id": "sign_separated_residual_data_role_preflight",
                        "status": "pending",
                    },
                ],
            },
            "records": [
                {
                    "id": "sign-separated-residual-repair-synthetic-prerequisite",
                    "terminal": "PASS",
                    "outcome": (
                        "identity_signed_routes_and_branch_isolation_contract_passed"
                    ),
                    "evidence": [
                        {
                            "path": str(audit_path.relative_to(root)),
                            "sha256": sha256_file(audit_path),
                        }
                    ],
                }
            ],
        }
        ledger_path = root / "docs/current-primary-quality-loop-ledger.json"
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return ledger_path

    def read_plan(self, root: Path) -> dict:
        return json.loads((root / ROLE_PLAN_PATH).read_text(encoding="utf-8"))

    def write_plan(self, root: Path, plan: dict) -> None:
        (root / ROLE_PLAN_PATH).write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_killed_family_cannot_reopen_data_role_preflight(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED", result)
        self.assertFalse(result["runnable"])
        self.assertIn("active iteration", result["reason"])

    def test_synthetic_fixture_passes(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PASS", result)
            self.assertEqual(result["overlap_count"], 0)
            self.assertFalse(result["training_cli_enabled"])
            self.assertEqual(result["pixel_decoder_imports"], [])

    def test_manifest_hash_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            manifest = root / "docs/scut-next120-nonoverlap-relative.txt"
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "extra.jpg\n",
                encoding="utf-8",
            )
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("manifest artifact hash mismatch", result["reason"])

    def test_reserved_blind_cannot_be_enabled(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            plan = self.read_plan(root)
            plan["roles"]["reserved_blind"]["authorized"] = True
            self.write_plan(root, plan)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("reserved blind", result["reason"])

    def test_missing_synthetic_authority_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["records"] = []
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("synthetic prerequisite PASS record", result["reason"])

    def test_training_cli_enablement_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            train_script = root / "scripts/train/train_patch_cleanup_erasemap_probe.py"
            train_script.write_text(
                train_script.read_text(encoding="utf-8")
                + "\n# sign_separated_residual_delta\n",
                encoding="utf-8",
            )
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("training CLI", result["reason"])

    def test_existing_target_patch_manifest_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            plan = self.read_plan(root)
            patch_path = root / plan["planned_outputs_must_be_absent"][
                "target_derived_patch_manifest"
            ]
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text("forbidden\n", encoding="utf-8")
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("planned output must be absent", result["reason"])

    def test_second_stage_hash_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            plan = self.read_plan(root)
            plan["baseline"]["current_second_stage"]["checkpoint"][
                "sha256"
            ] = "0" * 64
            self.write_plan(root, plan)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("artifact hash mismatch", result["reason"])

    def test_parent_traversal_manifest_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            plan = self.read_plan(root)
            plan["roles"]["inner_val15"]["manifest"]["path"] = "../outside.txt"
            self.write_plan(root, plan)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("must stay inside repository", result["reason"])


if __name__ == "__main__":
    unittest.main()
