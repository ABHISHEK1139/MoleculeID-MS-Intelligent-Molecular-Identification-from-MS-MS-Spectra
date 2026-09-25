"""Helper script to submit Kernel Version 12 as Submission 10 to Kaggle enveda-CASMI26.

Run this script as soon as the daily submission quota resets at 00:00:00 UTC.
"""
from __future__ import annotations

import subprocess
import sys
import time


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    import requests

    api = KaggleApi()
    api.authenticate()

    limits = api.competition_get_submission_limits("enveda-CASMI26-molecule-id-mass-spectra")
    num_today = getattr(limits, "num_today", getattr(limits, "_num_today", 0))
    allowed_now = getattr(limits, "num_allowed_now", getattr(limits, "_num_allowed_now", 0))
    print(f"Submission status for today: {num_today}/5 used ({allowed_now} remaining)")

    message = (
        "Submission 10 (Stage 6): Calibrated Candidate Router (tau=0.90) + "
        "Anti-False-Analog Gating + Stage 5 (W=2.00)"
    )

    if allowed_now <= 0:
        print("Daily submission quota (5/5) is currently exhausted.")
        print("Quota resets at 00:00:00 UTC (05:30:00 AM IST).")
        return

    print(f"Submitting Kernel Version 12 to Kaggle CASMI 2026: {message}...", flush=True)
    try:
        res = api.competition_submit_code(
            file_name="submission.csv",
            message=message,
            competition="enveda-CASMI26-molecule-id-mass-spectra",
            kernel="abhishek6545/casmi26-stage-6-submission",
            kernel_version=12,
        )
        print("Submission dispatched successfully! Response:", res)
        print("Waiting 20s to fetch public score...", flush=True)
        time.sleep(20)
        subs = api.competition_submissions("enveda-CASMI26-molecule-id-mass-spectra")
        if subs:
            latest = subs[0]
            print(f"Latest submission: Ref {latest.ref} | Status: {latest.status} | Public Score: {latest.public_score}")
    except requests.exceptions.HTTPError as e:
        print("HTTP ERROR:", e.response.status_code, e.response.text)
    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
