"""Monitor Kaggle Kernel Version 18, dispatch competition submission, and poll official leaderboard score."""
import time
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel_slug = "abhishek6545/casmi26-sota-meta-ranker"
    comp_slug = "enveda-CASMI26-molecule-id-mass-spectra"
    version_num = 20

    print(f"Monitoring Kaggle Kernel: {kernel_slug} (Target Version: {version_num})...", flush=True)
    start_time = time.time()
    last_status = None
    has_started = False

    while True:
        try:
            status_info = api.kernels_status(kernel_slug)
            status = status_info.get("status") if isinstance(status_info, dict) else getattr(status_info, "status", str(status_info))
            status_str = str(status).upper()
            elapsed_m = (time.time() - start_time) / 60.0

            if status_str != last_status:
                print(f"[{elapsed_m:.1f}m] Kernel status transition: {last_status} -> {status_str}", flush=True)
                last_status = status_str

            if any(s in status_str for s in ["RUNNING", "QUEUED"]):
                has_started = True

            if "COMPLETE" in status_str:
                print(f"\n[SUCCESS] Kernel run completed in {elapsed_m:.1f} minutes!")
                time.sleep(5)

                # Submit to competition
                message = "Submission 14: Validated Fixed Linear Fusion + InChIKey14 Library Match + FPNet (v20)"
                print(f"Submitting version {version_num} to competition: {comp_slug}...", flush=True)
                try:
                    res = api.competition_submit_code(
                        file_name="submission.csv",
                        message=message,
                        competition=comp_slug,
                        kernel=kernel_slug,
                        kernel_version=version_num,
                    )
                    print(f"[SUBMIT] Submission dispatched successfully: {res}", flush=True)
                except Exception as e:
                    print(f"[SUBMIT ERROR] {e}", flush=True)

                print("Waiting for competition score evaluation...", flush=True)
                sub_start = time.time()
                while time.time() - sub_start < 600:
                    time.sleep(15)
                    subs = api.competition_submissions(comp_slug)
                    if subs:
                        latest = subs[0]
                        score = latest.public_score
                        st = str(latest.status).upper()
                        print(f"  Submission Status: {st} | Public Score: {score} | Ref: {latest.ref}", flush=True)
                        if "COMPLETE" in st and score is not None:
                            print("\n" + "=" * 65)
                            print(f"  NEW OFFICIAL KAGGLE PUBLIC LEADERBOARD SCORE: {score}")
                            print("=" * 65, flush=True)
                            return
                return

            elif has_started and any(err in status_str for err in ["FAIL", "CANCEL", "ERROR"]):
                print(f"\n[FAILURE] Kernel run failed with status: {status_str}", flush=True)
                return

        except Exception as e:
            print(f"[POLL ERROR] {e}", flush=True)

        time.sleep(20)

if __name__ == '__main__':
    main()
