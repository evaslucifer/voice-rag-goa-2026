"""Direct runner script for the Voice-Enabled Multilingual RAG Backend."""

import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure backend directory is in sys.path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")

    print("\n" + "=" * 60)
    print("  🚀 Starting Voice-Enabled Multilingual RAG Backend")
    print(f"  📖 Docs (Swagger):  http://localhost:{port}/docs")
    print(f"  💓 Health Check:    http://localhost:{port}/api/health")
    print("=" * 60 + "\n")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        app_dir=backend_dir,
    )
