import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000/api"
IMAGE_PATH = "../image.png"

def test_pipeline():
    print("1. Uploading image to start analysis...")
    with open(IMAGE_PATH, "rb") as f:
        files = {"image": ("image.png", f, "image/png")}
        resp = requests.post(f"{BASE_URL}/analyze", files=files)
        
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"Task started: {task_id}")
    
    print("2. Listening to SSE stream...")
    # For testing, we can just do a GET request with stream=True and read lines
    with requests.get(f"{BASE_URL}/analyze/{task_id}/stream", stream=True) as sse_resp:
        for line in sse_resp.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data = json.loads(decoded_line[6:])
                    print(f"[{data.get('event')}] {data.get('module')}")
                    if data.get('event') == 'pipeline_complete':
                        print("Pipeline finished successfully!")
                        break
                    if data.get('event') == 'pipeline_error':
                        print("Pipeline error:", data.get('error'))
                        exit(1)
    
    print("3. Checking Database History...")
    history_resp = requests.get(f"{BASE_URL}/history")
    history_resp.raise_for_status()
    history = history_resp.json()
    assert any(t["id"] == task_id for t in history), "Task not found in history"
    print(f"History tracking works. Found {len(history)} total tasks.")
    
    print("4. Testing VQA (Ask a question)...")
    question = "What is the main subject of this image?"
    print(f"Q: {question}")
    vqa_resp = requests.post(f"{BASE_URL}/analyze/{task_id}/ask", json={"question": question})
    vqa_resp.raise_for_status()
    print(f"A: {vqa_resp.json().get('answer')}")
    
    print("All Phase 3 backend tests passed successfully!")

if __name__ == "__main__":
    test_pipeline()
