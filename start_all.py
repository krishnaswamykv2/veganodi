

import argparse
import os
import subprocess
import sys
import time

PYTHON = sys.executable 
IS_WINDOWS = os.name == "nt"


def wait_for_file(path, timeout_s=30, poll_s=0.5):
    """Blocks until `path` exists and has at least one line."""
    print(f"Waiting for {path} to start receiving data...")
    waited = 0.0
    while waited < timeout_s:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    if f.readline().strip():
                        print(f"  -> {path} is live.")
                        return True
            except OSError:
                pass
        time.sleep(poll_s)
        waited += poll_s
    print(f"  -> WARNING: {path} did not receive data within {timeout_s}s. "
          f"Continuing anyway — check your camera source.")
    return False


def wait_for_port(host, port, timeout_s=20, poll_s=0.5):
    """Blocks until something is listening on host:port — used to make sure
    the FastAPI server is actually up before the frontend needs it."""
    import socket
    print(f"Waiting for server on {host}:{port}...")
    waited = 0.0
    while waited < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"  -> server on port {port} is live.")
                return True
        except OSError:
            time.sleep(poll_s)
            waited += poll_s
    print(f"  -> WARNING: nothing responded on port {port} within {timeout_s}s. Continuing anyway.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Veganodi one-command full-stack launcher")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo",
                         help="demo = synthetic data, no camera; live = real camera via perception.py")
    parser.add_argument("--source", default="0", help="camera index or video file (--mode live only)")
    parser.add_argument("--no-show", action="store_true", help="perception.py without the live preview window (--mode live only)")
    parser.add_argument("--detections-file", default="detections.jsonl")
    parser.add_argument("--server-port", default="8000")
    parser.add_argument("--frontend-dir", default="frontend")
    parser.add_argument("--npm-cmd", default="npm", help="override if npm isn't on PATH — e.g. full path to npm.cmd")
    parser.add_argument("--no-dashboard", action="store_true",
                         help="also skip the Streamlit dashboard (off by default since the website replaces it)")
    parser.add_argument("--with-streamlit-dashboard", action="store_true",
                         help="also launch the older Streamlit dashboard alongside the website")
    args = parser.parse_args()

    processes = []

    def start(name, cmd, cwd=None, shell=False):
        print(f"Starting {name}: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        proc = subprocess.Popen(cmd, cwd=cwd, shell=shell)
        processes.append((name, proc))
        return proc

    def stop_all():
        print("\nShutting down all Veganodi processes...")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                proc.terminate()
        time.sleep(2)
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Force-killing {name}...")
                proc.kill()
        print("All processes stopped.")

    try:
      
        if args.mode == "live":
            perception_cmd = [PYTHON, "perception.py", "--source", args.source,
                               "--out", args.detections_file]
            if not args.no_show:
                perception_cmd.append("--show")
            start("perception.py", perception_cmd)
            wait_for_file(args.detections_file)

       
        server_cmd = [PYTHON, "-m", "uvicorn", "server:app",
                       "--host", "0.0.0.0", "--port", args.server_port]
        start("server (uvicorn)", server_cmd)
        wait_for_port("localhost", int(args.server_port))

       
        frontend_cmd = f'"{args.npm_cmd}" run dev -- --host'
        if not os.path.isdir(args.frontend_dir):
            print(f"WARNING: frontend directory '{args.frontend_dir}' not found — "
                  f"skipping the website. Pass --frontend-dir if it's named differently.")
        else:
            start("frontend (vite)", frontend_cmd, cwd=args.frontend_dir, shell=True)

        
        if args.mode == "demo":
            ci_cmd = [PYTHON, "core_intelligence.py", "--demo"]
        else:
            ci_cmd = [PYTHON, "core_intelligence.py", "--jsonl", args.detections_file]
        start("core_intelligence.py", ci_cmd)

       
        if args.with_streamlit_dashboard and not args.no_dashboard:
            start("dashboard.py (Streamlit)", [PYTHON, "-m", "streamlit", "run", "dashboard.py"])

        print("\n" + "=" * 60)
        print(f"Veganodi is running in --mode {args.mode}.")
        print(f"  Website:  http://localhost:5173  (check frontend output above for the exact URL)")
        print(f"  API:      http://localhost:{args.server_port}")
        print("Press Ctrl+C to stop everything.")
        print("=" * 60 + "\n")

        while True:
            time.sleep(1)
            for name, proc in list(processes):
                if proc.poll() is not None:
                    print(f"\nWARNING: {name} exited unexpectedly "
                          f"(code {proc.returncode}). Check its output above.")
                    processes.remove((name, proc))
            if not processes:
                print("All processes have exited. Shutting down launcher.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_all()


if __name__ == "__main__":
    main()
