"""Monitor the CASMI 2026 SOTA kernel on Kaggle and submit as soon as complete."""
import time
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel_slug = "abhishek6545/casmi26-sota-meta-ranker"
    comp_slug = "enveda-CASMI26-molecule-id-mass-spectra"

    print(f"Monitoring Kaggle Kernel: {kernel_slug}...")
    start_time = time.time()
    last_status = None

    while True:
        status_info = api.kernels_status(kernel_slug)
        status = status_info.get("status") if isinstance(status_info, dict) else getattr(status_info, "status", str(status_info))
        failure_msg = status_info.get("failureMessage") if isinstance(status_info, dict) else getattr(status_info, "failure_message", None)

        elapsed = time.time() - start_time
        if status != last_status:
            print(f"[{elapsed:.0f}s] Kernel status transition: {last_status} -> {status}", flush=True)
            last_status = status

        if status == "COMPLETE":
            print(f"\n[SUCCESS] Kernel run completed in {elapsed:.0f}s! Dispatched output generation.")
            # Fetch latest version number
            time.sleep(5)
            # Submit to competition
            message = "CASMI 2026 SOTA v7: Quad-Channel + 1.38M Unified Lib + Neural FPNet Ensemble + MetFrag"
            print(f"Submitting to competition: {comp_slug} with message: '{message}'...")
            try:
                res = api.competition_submit_code(
                    file_name="submission.csv",
                    message=message,
                    competition=comp_slug,
                    kernel=kernel_slug,
                )
                print(f"[SUBMIT] Dispatched submission: {res}")
            except Exception as e:
                print(f"[SUBMIT ERROR] {e}")

            print("Waiting for competition score evaluation...", flush=True)
            for _ in range(12):
                time.sleep(15)
                subs = api.competition_submissions(comp_slug)
                if subs:
                    latest = subs[0]
                    print(f"  Submission Status: {latest.status} | Public Score: {latest.public_score} | Ref: {latest.ref}")
                    if str(latest.status).upper() in ["COMPLETE", "SUBMISSIONSTATUS.COMPLETE"] and latest.public_score is not None:
                        print(f"\n=======================================================")
                        print(f"  NEW SOTA PUBLIC LEADERBOARD SCORE: {latest.public_score}")
                        print(f"=======================================================")
                        return
            return

        elif status in ["FAILED", "CANCELLED", "ERROR"]:
            print(f"\n[FAILURE] Kernel run failed with status: {status}")
            print(f"Failure message: {failure_msg}")
            # Try fetching log
            try:
                log_info = api.kernel_output(kernel_slug)
                print("Log output:")
                print(log_info)
            except Exception as err:
                print(f"Could not retrieve logs: {err}")
            return

        time.sleep(20)

if __name__ == '__main__':
    main()
