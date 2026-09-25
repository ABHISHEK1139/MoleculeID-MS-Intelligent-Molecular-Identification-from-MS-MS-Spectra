"""Live monitor for Kaggle CASMI 2026 Submission Ref 56538881."""
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    comp = 'enveda-CASMI26-molecule-id-mass-spectra'
    target_ref = 56538881
    
    print(f"Starting live monitoring for submission Ref: {target_ref}...")
    start_time = time.time()
    
    while True:
        try:
            subs = api.competition_submissions(comp)
            target = next((s for s in subs if s.ref == target_ref), None)
            if target is None and subs:
                target = subs[0]
                
            elapsed_min = (time.time() - start_time) / 60.0
            status_str = str(getattr(target, 'status', '')).upper()
            score = getattr(target, 'public_score', None)
            
            print(f"[{elapsed_min:.1f}m] Ref: {target.ref} | Status: {status_str} | Public Score: {score}", flush=True)
            
            if "COMPLETE" in status_str:
                print("\n" + "=" * 65)
                print(f"  OFFICIAL PUBLIC LEADERBOARD SCORE: {score}")
                print("=" * 65, flush=True)
                break
            elif "ERROR" in status_str or "FAIL" in status_str:
                print(f"\n[ALERT] Submission ended with failure status: {status_str}", flush=True)
                break
                
        except Exception as e:
            print(f"[POLL ERROR] {e}", flush=True)
            
        time.sleep(30)

if __name__ == '__main__':
    main()
