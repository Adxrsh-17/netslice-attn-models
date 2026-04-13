import os

# Backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Storage directories
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
PDF_DIR = os.path.join(BACKEND_DIR, "assets", "papers", "references")
ARCH_PAPERS_DIR = os.path.join(BACKEND_DIR, "assets", "papers", "architecture")
PLOTS_DIR = os.path.join(BACKEND_DIR, "assets", "plots")
RESULTS_DIR = os.path.join(BACKEND_DIR, "assets", "results")
