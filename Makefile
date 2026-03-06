# Makefile -- One-word shortcuts for the MNIST pipeline.
#
# Usage:
#   make train       Train with config.py defaults (simple_fc, 5 epochs)
#   make cnn         Train CNN model
#   make quick       Smoke test: 1 epoch, fast feedback
#   make viz         Visualize existing model (skip training)
#   make all         Full pipeline: train + visualize
#   make eval        Evaluate existing model (no training)
#   make dry-run     Show pipeline plan without running
#   make clean       Remove __pycache__ dirs
#   make help        Show this help

PYTHON_PATH = /c/Users/shawn/miniconda3/envs/MachineLearning/python.exe

.PHONY: train cnn quick viz all eval dry-run clean help

help:
	@echo "MNIST Pipeline Shortcuts"
	@echo "========================"
	@echo "  make train   Train simple_fc (5 epochs)"
	@echo "  make cnn     Train CNN model (5 epochs)"
	@echo "  make quick   Smoke test (1 epoch)"
	@echo "  make viz     Visualize only (skip training)"
	@echo "  make all     Full pipeline (train + viz)"
	@echo "  make eval    Evaluate existing model (no training)"
	@echo "  make dry-run Show plan without running"
	@echo "  make clean   Remove __pycache__ dirs"

train:
	$(PYTHON_PATH) run_all.py

cnn:
	$(PYTHON_PATH) run_all.py --model cnn

quick:
	$(PYTHON_PATH) run_all.py --epochs 1

viz:
	$(PYTHON_PATH) run_all.py --skip-train

all:
	$(PYTHON_PATH) run_all.py

eval:
	$(PYTHON_PATH) run_all.py --eval-only

dry-run:
	$(PYTHON_PATH) run_all.py --dry-run

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
