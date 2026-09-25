"""Automated Dispatch & Monitor for Submission 12 (Clean Baseline).

Workflow:
1. Verify remaining Kaggle submission quota.
2. Push self-contained kernel 'abhishek6545/casmi26-stage-6-submission' to Kaggle.
3. Monitor kernel execution until status is 'COMPLETE'.
4. Submit kernel output to 'enveda-CASMI26-molecule-id-mass-spectra'.
5. Poll and report official Public Leaderboard score.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    print("=" * 85)
    print("  CASMI 2026: DISPATCHING SUBMISSION 12 (CLEAN BASELINE)")
    print("=" * 85, flush=True)

    api = KaggleApi()
    api.authenticate()

    competition = "enveda-CASMI26-molecule-id-mass-spectra"
    kernel_slug = "abhishek6545/casmi26-clean-baseline-submission-12"
    kernel_dir = Path("kaggle_submission_kernel")

    # 1. Verify Submission Quota
    limits = api.competition_get_submission_limits(competition)
    num_today = getattr(limits, "num_today", getattr(limits, "_num_today", 0))
    allowed_now = getattr(limits, "num_allowed_now", getattr(limits, "_num_allowed_now", 0))
    print(f"\n[1/4] Checking Daily Submission Quota: {num_today}/5 used ({allowed_now} remaining)...", flush=True)
    if allowed_now <= 0:
        print("ERROR: Daily submission quota exhausted. Resets at 00:00:00 UTC.")
        return

    # 2. Push Kernel
    print(f"\n[2/4] Pushing Kernel from {kernel_dir} to Kaggle ({kernel_slug})...", flush=True)
    res = api.kernels_push(str(kernel_dir))
    print("Kernel push initiated. Response:", res, flush=True)
    ver_num = getattr(res, "version_number", getattr(res, "versionNumber", None))
    if ver_num is None and isinstance(res, dict):
        ver_num = res.get("versionNumber") or res.get("version_number")
    print(f"Pushed Kernel Version: {ver_num}")

    # 3. Monitor Kernel Execution
    print("\n[3/4] Monitoring Kernel Execution on Kaggle CPU...", flush=True)
    t_start = time.time()
    last_status = ""
    for attempt in range(60):
        time.sleep(15)
        try:
            k_status = api.kernels_status(kernel_slug)
            s_str = str(getattr(k_status, "status", getattr(k_status, "_status", str(k_status)))).upper()
            if s_str != last_status:
                print(f"  [{int(time.time() - t_start)}s] Kernel Status: {s_str}", flush=True)
                last_status = s_str

            if "COMPLETE" in s_str:
                print(f"Kernel execution completed successfully in {int(time.time() - t_start)}s!")
                break
            elif "ERROR" in s_str or "FAIL" in s_str:
                print(f"ERROR: Kernel execution failed with status: {s_str}")
                return
        except Exception as e:
            print(f"  Notice during status poll: {e}", flush=True)

    # 4. Submit Kernel Version as Official Competition Submission
    message = "Submission 12: Clean Baseline (Fixed Binning, True Parity, Gated Library Bonus, Unique Top-25)"
    print(f"\n[4/4] Submitting Kernel Output to {competition}...", flush=True)
    print(f"Message: {message}", flush=True)

    try:
        sub_kwargs = {
            "file_name": "submission.csv",
            "message": message,
            "competition": competition,
            "kernel": kernel_slug,
        }
        if ver_num is not None:
            sub_kwargs["kernel_version"] = int(ver_num)

        sub_res = api.competition_submit_code(**sub_kwargs)
        print("Submission dispatched successfully! Response:", sub_res, flush=True)
    except Exception as e:
        print("Notice during competition_submit_code:", e, flush=True)

    # 5. Monitor Public Leaderboard Score
    print("\nPolling competition submission status for official Public Score...", flush=True)
    for attempt in range(30):
        time.sleep(15)
        try:
            subs = api.competition_submissions(competition)
            if subs:
                latest = subs[0]
                status = getattr(latest, "status", "unknown")
                score = getattr(latest, "public_score", None)
                print(f"  Submission Ref {latest.ref} | Status: {status} | Public Score: {score}", flush=True)
                if status.lower() == "complete" or score is not None:
                    print("\n" + "=" * 85)
                    print(f"  OFFICIAL PUBLIC LEADERBOARD SCORE: {score}")
                    print("=" * 85, flush=True)
                    break
        except Exception as e:
            print(f"  Notice while fetching score: {e}", flush=True)


if __name__ == "__main__":
    main()
