#!/usr/bin/env python3
import subprocess
import os
import sys
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")

    print("Starting Sahara Research Visualizer...")

    # Automatically run npm install if node_modules doesn't exist
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("-> node_modules not found. Running 'npm install' first... (this might take a moment)")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            print("-> npm install complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run npm install: {e}")
            return

    # Use buffering=1 for line buffering so logs flush immediately
    backend_log = open("backend_startup.log", "w", buffering=1)
    frontend_log = open("frontend_startup.log", "w", buffering=1)

    print("-> Starting FastAPI backend (logging to backend_startup.log)...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=backend_dir,
        stdout=backend_log,
        stderr=subprocess.STDOUT
    )

    print("-> Starting Vite frontend (logging to frontend_startup.log)...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        stdout=frontend_log,
        stderr=subprocess.STDOUT
    )

    print("\n✅ Both servers are running in the background!")
    print("🌐 Web UI available at: http://localhost:5173")
    print("\nPress Ctrl+C to shut down both servers.")

    try:
        while True:
            time.sleep(1)
            
            b_status = backend_process.poll()
            if b_status is not None:
                print(f"\n❌ Backend process died unexpectedly (exit code {b_status})! Check backend_startup.log")
                break
                
            f_status = frontend_process.poll()
            if f_status is not None:
                print(f"\n❌ Frontend process died unexpectedly (exit code {f_status})! Check frontend_startup.log")
                break

    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        
    finally:
        if backend_process.poll() is None:
            backend_process.terminate()
        if frontend_process.poll() is None:
            frontend_process.terminate()
            
        backend_log.close()
        frontend_log.close()
        print("Done. Goodbye!")

if __name__ == "__main__":
    main()
