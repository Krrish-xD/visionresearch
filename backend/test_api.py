import httpx
import json
import time

def test_backend():
    print("Starting backend test...")
    with open("../image.png", "rb") as f:
        # 1. Start Analysis
        print("Uploading image...")
        response = httpx.post("http://127.0.0.1:8000/api/analyze", files={"image": f})
        response.raise_for_status()
        data = response.json()
        task_id = data["task_id"]
        print(f"Task ID: {task_id}")
        
        # 2. Connect to SSE stream
        print(f"Connecting to stream for task {task_id}...")
        with httpx.stream("GET", f"http://127.0.0.1:8000/api/analyze/{task_id}/stream", timeout=300.0) as stream:
            for line in stream.iter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    event_type = payload.get("event")
                    module = payload.get("module")
                    
                    if event_type == "pipeline_start":
                        print(f"✅ PIPELINE STARTED! Total modules: {payload['total_stages']}")
                    elif event_type == "module_start":
                        print(f"  -> Starting module: {module}...")
                    elif event_type == "module_complete":
                        print(f"  ✅ Finished module: {module} in {payload.get('timing_ms')}ms")
                        if "results" in payload:
                            print(f"     Results keys: {list(payload['results'].keys())}")
                    elif event_type == "module_error":
                        print(f"  ❌ ERROR in module {module}: {payload.get('error')}")
                    elif event_type == "pipeline_complete":
                        print(f"✅ PIPELINE COMPLETE in {payload.get('total_time_ms')}ms!")
                        break
                    elif event_type == "pipeline_error":
                        print(f"❌ PIPELINE ERROR: {payload.get('error')}")
                        break
                    elif event_type == "memory_update":
                        pass # Ignore spam
                    else:
                        print(f"Unknown event: {event_type}")

if __name__ == "__main__":
    # Wait a bit in case the backend was just restarted
    time.sleep(2)
    test_backend()
