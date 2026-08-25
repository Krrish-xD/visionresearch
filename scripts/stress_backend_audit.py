"""
Empirical Backend & Singleton Stress Harness for VisionResearch.
Runs comprehensive stress tests against FastAPI backend and VLMEngine lifecycle.
"""

import sys
import os
import time
import json
import io
import socket
import subprocess
import requests
import concurrent.futures
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from src.vlm.engine import VLMEngine

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def make_dummy_png(w=10, h=10, color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, "PNG")
    return buf.getvalue()

def run_stress_audit():
    print("=" * 70)
    print("VISIONRESEARCH BACKEND & SINGLETON ADVERSARIAL STRESS AUDIT")
    print("=" * 70)

    port = find_free_port()
    print(f"[1/6] Spawning live Uvicorn backend on port {port}...")
    
    proc = subprocess.Popen(
        ["venv/bin/python", "-m", "uvicorn", "backend.main:app", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        # Wait for server readiness
        server_ready = False
        for _ in range(40):
            try:
                r = requests.get(f"{base_url}/api/models", timeout=1)
                if r.status_code == 200:
                    server_ready = True
                    break
            except Exception:
                time.sleep(0.25)

        if not server_ready:
            print("FATAL: Backend server failed to start within timeout.")
            return

        print("Backend server is UP and responding to requests.")

        # ===================================================================
        # TEST 1: Rapid Concurrent Requests
        # ===================================================================
        print("\n[2/6] Running Concurrency Stress Test (200 requests)...")
        num_concurrent = 200
        latencies = []
        status_codes = []

        def hit_endpoint(i):
            url = f"{base_url}/api/models" if i % 2 == 0 else f"{base_url}/api/model/status"
            t0 = time.time()
            resp = requests.get(url, timeout=5)
            elapsed = time.time() - t0
            return resp.status_code, elapsed

        t_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(hit_endpoint, i) for i in range(num_concurrent)]
            for f in concurrent.futures.as_completed(futures):
                code, el = f.result()
                status_codes.append(code)
                latencies.append(el)
        total_time = time.time() - t_start

        success_count = sum(1 for c in status_codes if c == 200)
        print(f"  -> Total requests: {num_concurrent}")
        print(f"  -> Success rate: {success_count}/{num_concurrent} ({success_count/num_concurrent*100:.1f}%)")
        print(f"  -> Total elapsed time: {total_time:.3f}s (RPS: {num_concurrent/total_time:.1f})")
        print(f"  -> Latency: Min={min(latencies)*1000:.1f}ms, Mean={sum(latencies)/len(latencies)*1000:.1f}ms, Max={max(latencies)*1000:.1f}ms")

        # ===================================================================
        # TEST 2: Invalid HTTP Methods
        # ===================================================================
        print("\n[3/6] Running Invalid HTTP Methods Fuzzing...")
        methods = ["post", "put", "delete", "patch"]
        endpoints = ["/api/models", "/api/model/status"]
        for ep in endpoints:
            for m in methods:
                r = getattr(requests, m)(f"{base_url}{ep}")
                print(f"  -> {m.upper()} {ep} => Status {r.status_code} (Expected 405: {'PASS' if r.status_code == 405 else 'FAIL'})")

        for m in ["get", "put", "delete", "patch"]:
            r = getattr(requests, m)(f"{base_url}/api/model/load")
            print(f"  -> {m.upper()} /api/model/load => Status {r.status_code} (Expected 405: {'PASS' if r.status_code == 405 else 'FAIL'})")

        # ===================================================================
        # TEST 3: Malformed & Adversarial Payloads
        # ===================================================================
        print("\n[4/6] Running Malformed Payloads & Injection Tests...")
        # 1. Invalid JSON
        r = requests.post(f"{base_url}/api/model/load", data="{bad_json:", headers={"Content-Type": "application/json"})
        print(f"  -> Broken JSON payload => Status {r.status_code} (Expected 422: {'PASS' if r.status_code == 422 else 'FAIL'})")

        # 2. Empty JSON
        r = requests.post(f"{base_url}/api/model/load", json={})
        print(f"  -> Empty JSON payload => Status {r.status_code} (Expected 422: {'PASS' if r.status_code == 422 else 'FAIL'})")

        # 3. Wrong data types
        r = requests.post(f"{base_url}/api/model/load", json={"model_id": 99999})
        print(f"  -> Integer model_id => Status {r.status_code} (Expected 422: {'PASS' if r.status_code == 422 else 'FAIL'})")

        # 4. Path traversal / Injection strings
        adversarial_inputs = [
            "../../../../etc/passwd",
            "/dev/null",
            "model\x00hidden",
            "A" * 10000
        ]
        for adv in adversarial_inputs:
            r = requests.post(f"{base_url}/api/model/load", json={"model_id": adv})
            print(f"  -> Adversarial ID '{adv[:20]}...' => Status {r.status_code} (Server survived: {'PASS' if r.status_code in (422, 500) else 'FAIL'})")

        # ===================================================================
        # TEST 4: Corrupted & Adversarial Media Inputs
        # ===================================================================
        print("\n[5/6] Running Corrupted Image & Media Input Tests...")
        corrupt_inputs = [
            ("0-byte empty file", b""),
            ("Truncated PNG header", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
            ("64KB random binary noise", os.urandom(65536)),
            ("Fake PDF disguised as PNG", b"%PDF-1.4\n1 0 obj\n<< /Title (Exploit) >>\nendobj"),
        ]

        # Let's test how generate handles corrupt images
        for label, data in corrupt_inputs:
            r = requests.post(
                f"{base_url}/api/generate",
                data={"prompt": "Test prompt", "model_id": "nonexistent_model"},
                files={"image": ("input.png", data, "image/png")}
            )
            print(f"  -> Corrupt input '{label}' => Status {r.status_code}, Response: {r.json().get('detail', '')[:60]}...")

        # ===================================================================
        # TEST 5: VLMEngine State Corruption Verification
        # ===================================================================
        print("\n[6/6] Verifying VLMEngine Singleton Lifecycle & State Corruption Bug...")
        
        # Test direct VLMEngine state corruption
        eng = VLMEngine()
        eng.model = "mock_model"
        eng.processor = "mock_proc"
        eng.current_model_id = "mock_id"
        print(f"  -> Initial state: hasattr(model)={hasattr(eng, 'model')}, hasattr(processor)={hasattr(eng, 'processor')}")

        try:
            eng.load_model("definitely_nonexistent_model_path_xyz")
        except Exception as e:
            print(f"  -> Expected load failure occurred: {type(e).__name__}")

        model_attr_exists = hasattr(eng, "model")
        proc_attr_exists = hasattr(eng, "processor")
        print(f"  -> State after failed load: hasattr(model)={model_attr_exists}, hasattr(processor)={proc_attr_exists}")
        
        if not model_attr_exists:
            print("  -> BUG CONFIRMED: `del self.model` removed attribute from VLMEngine instance!")
            try:
                _ = eng.model is not None
            except AttributeError as ae:
                print(f"     Accessing eng.model raises AttributeError: '{ae}'")
            try:
                eng.load_model("another_path")
            except AttributeError as ae:
                print(f"     Subsequent eng.load_model crashes on AttributeError: '{ae}'")

    finally:
        print("\n[Cleanup] Terminating Uvicorn backend process...")
        proc.terminate()
        proc.wait(timeout=5)
        print("Backend process terminated and cleaned up cleanly.")
        print("=" * 70)
        print("STRESS AUDIT COMPLETED.")
        print("=" * 70)

if __name__ == "__main__":
    run_stress_audit()
