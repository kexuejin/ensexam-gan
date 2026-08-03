# HW5K candidate blind test set

This directory contains the immutable HW5K official test set that was consumed by
the reportable current-primary final-blind evaluation on 2026-07-26. It is no
longer eligible as a fresh blind set for Candidate 5 or any later model. It must
never be used for training, validation, calibration, threshold selection,
qualitative tuning, or any experiment whose outcome influences model selection.
See `docs/decisions/2026-07-26-hw5k-final-blind-current-primary.md`.

## Origin

- Supplied archive: `/Volumes/Kapp/tmp/ensexam-hw5k-reserved-intake-20260726/HW5K.7z`
- Archive subtrees registered: `HW5K/test/input` and `HW5K/test/target`
- Download date: 2026-07-26. The archive was downloaded into the intake directory during this session; this date is also reflected by its UTC filesystem timestamp (`2026-07-26T02:39:08Z`).
- Archive size: 9,309,694,714 bytes
- Archive SHA-256: `326e7179ec46f3be1da2a5f0a99afff03e9337c7851ef22a09aae705f00ee952`, matching the Hugging Face LFS object ID observed before download.

## Registered layout

`reserved/` contains only these two real directories (no symbolic links):

- `all_images/` — 525 original test input files, 896,301,849 bytes total
- `all_labels/` — 525 original test target files, 816,279,444 bytes total

There is no `masks/` directory, no predictions, and no other generated artifact in `reserved/`.

## Pairing and integrity rule

Each entry is a regular `.jpg`, `.jpeg`, or `.png` file named by a numeric ID. Every numeric ID occurs exactly once in `all_images/` and exactly once in `all_labels/`; the two sorted ID sets are identical. The copied tree was verified against the extracted source with a bytewise recursive comparison (`diff -qr`) on 2026-07-26.

## Registration state

The initial 2026-07-26 preflight validated the physical layout, provenance
declaration, pair matching, and absence of masks. Its name-only overlap scan
reported generic numeric filename collisions. A later content-identity-aware
isolation path admitted and registered all 525 samples under
`outputs/hw5k_reserved_blind_registration_20260726/`. Current-primary inference,
post-freeze scoring, and completion verification then ran under
`outputs/hw5k_final_blind_primary_current_primary_20260726/`.

That one permitted final-blind use is complete. Preserve the files for audit and
reproduction only. Do not use their inputs, labels, metrics, or visual results to
alter a later model, preprocessing, metrics, thresholds, selection procedure, or
promotion decision. A later candidate requires a separately reserved, unseen
final test set.
