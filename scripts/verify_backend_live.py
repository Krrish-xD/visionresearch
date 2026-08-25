#!/usr/bin/env python3
"""Independent Victory Audit Backend Live Verification Script."""

import subprocess
import time
import sys
import json
import urllib.request
import urllib.error
import os
import signal

def run_live_backend_test():
    print("[1/5] Starting FastAPI server on port 8000...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8000"],
        cwd="/home/kxd/Projects/visionresearch",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    server_url = "http://127.0.0.1:8000"
    max_retries = 20
    is_ready = False

    print("[2/5] Waiting for server to become ready...")
    for i in range(max_retries):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"{server_url}/docs", timeout=2) as resp:
                if resp.status == 200:
                    is_ready = True
                    print(f"Server is up and responsive after {(i+1)*0.5:.1f}s.")
                    break
        except Exception:
            pass

    if not is_ready:
        print("ERROR: Server failed to start within timeout.")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        out, err = proc.communicate(timeout=5)
        print(f"STDOUT: {out.decode()}")
        print(f"STDERR: {err.decode()}")
        sys.exit(1)

    try:
        print("[3/5] Testing GET /api/models...")
        with urllib.request.urlopen(f"{server_url}/api/models", timeout=5) as resp:
            status_code = resp.status
            body = resp.read().decode()
            data = json.loads(body)
            print(f"  Status: {status_code}")
            print(f"  Response: {data}")
            assert status_code == 200, f"Expected status 200, got {status_code}"
            assert isinstance(data, list), f"Expected list response, got {type(data)}"

        print("[4/5] Testing GET /api/model/status...")
        with urllib.request.urlopen(f"{server_url}/api/model/status", timeout=5) as resp:
            status_code = resp.status
            body = resp.read().decode()
            data = json.loads(body)
            print(f"  Status: {status_code}")
            print(f"  Response: {data}")
            assert status_code == 200, f"Expected status 200, got {status_code}"
            assert "active_model_id" in data, "Missing active_model_id key"
            assert "is_loaded" in data, "Missing is_loaded key"
            assert data["is_loaded"] is False, f"Expected is_loaded=False initially, got {data['is_loaded']}"
            assert data["active_model_id"] is None, f"Expected active_model_id=None initially, got {data['active_model_id']}"

        print("All endpoint assertions PASSED!")

    finally:
        print("[5/5] Terminating FastAPI server cleanly...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        out, err = proc.communicate(timeout=5)
        print(f"Server exited with returncode {proc.returncode}.")
        print("Clean termination confirmed.")

if __name__ == "__main__":
    run_live_backend_test()
