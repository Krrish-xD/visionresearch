#!/usr/bin/env python3
import subprocess
import os
import sys
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")

    # 1. Ensure Python virtual environment exists and dependencies are installed
    venv_dir = os.path.join(root_dir, "venv")
    if not os.path.exists(venv_dir):
        print("-> venv not found. Creating virtual environment and installing dependencies... (this might take a moment)")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=root_dir, check=True)
            venv_pip = os.path.join(venv_dir, "bin", "pip")
            if os.name == 'nt':
                venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
            subprocess.run([venv_pip, "install", "-r", "requirements.txt"], cwd=root_dir, check=True)
            print("-> venv setup and pip install complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup virtual environment or install dependencies: {e}")
            return

    # Automatically use the virtual environment's python
    venv_python = os.path.join(venv_dir, "bin", "python")
    if os.name == 'nt':
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    
    python_executable = venv_python if os.path.exists(venv_python) else sys.executable

    print("Starting Vision Research Visualizer...")

    # 2. Ensure frontend dependencies are installed
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("-> node_modules not found. Running 'npm install' first... (this might take a moment)")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            print("-> npm install complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run npm install: {e}")
            return

    # 3. Setup and verify all resources (datasets, splits) via unified setup_resources.py
    setup_script = os.path.join(root_dir, "scripts", "setup_resources.py")
    if os.path.exists(setup_script):
        try:
            subprocess.run([python_executable, setup_script], cwd=root_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Resource setup encountered an error: {e}")

    # 4. Launch backend and frontend servers
    backend_log = open("backend_startup.log", "w", buffering=1)
    frontend_log = open("frontend_startup.log", "w", buffering=1)

    print("-> Starting FastAPI backend (logging to backend_startup.log)...")
    backend_process = subprocess.Popen(
        [python_executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
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
