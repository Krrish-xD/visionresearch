import asyncio
import httpx
import sys

async def run_test():
    image_path = "/home/kxd/Coding/visionresearch/image.png"
    url = "http://127.0.0.1:8000/api/analyze"
    
    print(f"Uploading {image_path} to {url}...")
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(url, files=files)
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"Success! Task ID: {data['task_id']}")
                    
                    # Connect to SSE
                    stream_url = f"{url}/{data['task_id']}/stream"
                    print(f"Listening to SSE at {stream_url}")
                    async with client.stream("GET", stream_url) as stream:
                        async for line in stream.aiter_lines():
                            if line.startswith("data: "):
                                print(line[6:])
                else:
                    print(response.text)
    except Exception as e:
        print(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test())
