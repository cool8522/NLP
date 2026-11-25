"""Start the local dashboard: python run.py [--port 8000]."""
import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("paperlab.api:app", host="127.0.0.1", port=args.port)

