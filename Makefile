# Makefile -- One-word shortcuts for the MNIST pipeline.
#
# Usage:
#   make train       Train with config.py defaults (simple_fc, 5 epochs)
#   make cnn         Train CNN model
#   make quick       Smoke test: 1 epoch, fast feedback
#   make viz         Visualize existing model (skip training)
#   make all         Full pipeline: train + visualize
#   make eval        Run detailed evaluation only
#   make clean       Remove __pycache__ dirs
#   make help        Show this help

PYTHON = /c/Users/shawn/miniconda3/envs/MachineLearning/python.exe

.PHONY: train cnn quick viz all eval clean help

help:
	@echo "MNIST Pipeline Shortcuts"
	@echo "========================"
	@echo "  make train   Train simple_fc (5 epochs)"
	@echo "  make cnn     Train CNN model (5 epochs)"
	@echo "  make quick   Smoke test (1 epoch)"
	@echo "  make viz     Visualize only (skip training)"
	@echo "  make all     Full pipeline (train + viz)"
	@echo "  make eval    Detailed evaluation only"
	@echo "  make clean   Remove __pycache__ dirs"

train:
	$(PYTHON) run_all.py

cnn:
	$(PYTHON) run_all.py --model cnn

quick:
	$(PYTHON) run_all.py --epochs 1

viz:
	$(PYTHON) run_all.py --skip-train

all:
	$(PYTHON) run_all.py

eval:
	$(PYTHON) run_all.py --skip-viz

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
