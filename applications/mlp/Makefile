# AxCept-Bench / AxAccept-Bench -- GENANN kernel + digits workload
# Author: Guilherme Saides Serbai (UTFPR Campo Mourao)
# Year: 2026
#
# `make` alone does EVERYTHING that can be automated (check/install
# scikit-learn, export the dataset, build host and riscv). The experiment
# itself runs under AxPike and is started manually:
#
#     make init_weights
#     python3 train_sweep.py
#     python3 validate_matrix.py
#
# The host-trained model.bin (`make train`) is optional: the reference
# model is the one trained under AxPike at fault probability 0.
#
# Project layout (see README.md section "Mapa de arquivos" for the full
# details of each one):
#
#   lib/                        -- GENANN library, do not touch
#   dataset/                    -- export of the digits dataset (host)
#   train_axram/                -- TRAINS the network under AxPike/AxRAM
#   train_exact_and_validate/   -- TRAINS exact (host) + VALIDATES under AxPike/AxRAM
#
# All compiled binaries and all artifacts generated at runtime
# (dataset_digits/, model.bin, init_weights.bin, the CSVs/logs) stay in the
# project ROOT, not inside the subdirectories -- the subdirectories only organize
# the SOURCE CODE. Always run the Python scripts and the binaries from the
# root (`kernel/genann/`).
#
# Architecture of the two builds:
#   host   -- native, exact. Trains (train_exact_and_validate/), generates initial
#             weights and evaluates models (train_axram/, train_exact_and_validate/).
#             Nothing here runs under AxPike/AxRAM.
#   riscv  -- cross build of the two KERNELS that run under AxPike: the
#             VALIDATION/inference one (train_exact_and_validate/) and the
#             TRAINING one (train_axram/).
#
# -fpermissive in the riscv build is NOT optional: g++ treats .c as C++ (that
# is why kiss_fft.c compiles in a single line with the other kernels), and
# genann.c has two malloc() calls without an explicit cast (valid in C,
# an error in strict C++). The flag lowers that to a warning without changing
# genann.c -- already tested (build + run, output matching bit for bit).

# -DNDEBUG (riscv code only, NOT on the host): disables assert() only in the build
# that runs under AxPike/AxRAM. lib/genann.c has `assert(!isnan(a))` in the sigmoid
# (line 75) -- meant to catch real bugs on the host, but under AxRAM a
# single bit-flip in a weight turns into NaN and that assert aborts the guest
# process immediately, without saving anything (training without checkpoints -- see
# README section 9). Training under AxRAM must continue even with corrupted data,
# instead of aborting -- -DNDEBUG does exactly that (assert() becomes a no-op when
# NDEBUG is defined, the standard <assert.h> behavior), without touching
# lib/genann.c (still "do not touch" -- only the compile FLAG changes, and only
# for the riscv side). The host keeps the assert enabled, so a real bug
# on the host still aborts and reports it.
PYTHON   = python3
CC       = gcc
RV_CXX   = riscv64-unknown-elf-g++
RV_FLAGS = -O3 -static -fpermissive -DNDEBUG

LIB_DIR   = lib
LIB_FLAGS = -I$(LIB_DIR)
LIB_SRC   = $(LIB_DIR)/genann.c
LIB_HDR   = $(LIB_DIR)/genann.h

TRAIN_AXRAM_DIR = train_axram
TRAIN_VAL_DIR   = train_exact_and_validate

# FIXED architecture of the experiment: 96 units, 2 hidden layers
# (16,522 weights, ~132KB). This is the ONLY configuration used to generate the
# reported results -- there is no second architecture competing for the
# same results space, precisely to avoid any reading of
# "we picked the one that worked". The choice was motivated by an earlier
# sizing pilot: with a much smaller network (32 units, 1 layer,
# 2,410 weights = ~19KB), the weight buffer fits almost entirely in the simulated L2
# (1024KB) together with the training dataset (1437 samples x 74 doubles x 8
# bytes = ~850KB), so most accesses after the first epoch
# become cache HITs -- mem_read_prob only affects reads that MISS the L2 and go
# to DRAM. In that pilot, the lowest error rates produced weights
# byte-for-byte identical to the noise-free baseline, evidence that no
# real bit-flip happened during the whole training in that
# configuration. The 96/2 architecture used here has a working set large
# enough to expose the network to noise measurably at all
# tested rates (see README section 9 and the paper for the full
# results). Accepted trade-off: slower simulation per epoch (~200s vs.
# ~10.6s for the 32/1 pilot) -- adjust EPOCHS below/in train_sweep.py if the
# total sweep time gets too high.
HIDDEN_UNITS  = 80
HIDDEN_LAYERS = 2
EPOCHS        = 300
INIT_SEED     = 42

.PHONY: all setup dataset host riscv train init_weights clean distclean

all: setup dataset host riscv
	@echo ""
	@echo "=========================================================="
	@echo "Tudo pronto: dataset_digits/ e binarios host + riscv ok."
	@echo "Falta so rodar o experimento sob o AxPike:"
	@echo ""
	@echo "    make init_weights"
	@echo "    python3 train_sweep.py"
	@echo "    python3 validate_matrix.py"
	@echo "=========================================================="

# Checks whether scikit-learn is already installed before touching anything -- only
# installs it if it is really missing.
setup:
	@echo "Checando scikit-learn..."
	@$(PYTHON) -c "import sklearn" 2>/dev/null && \
		echo "scikit-learn ja instalado, mantendo." || ( \
		echo "scikit-learn nao encontrado, instalando..." && \
		( $(PYTHON) -m pip install --user scikit-learn || \
		  $(PYTHON) -m pip install --user --break-system-packages scikit-learn ) \
	)

# Downloads (via sklearn) and exports the `digits` dataset to dataset_digits/.
dataset: setup
	$(PYTHON) dataset/export_digits.py

# train_exact_host      -- the real trainer (host, exact), reads
#   dataset_digits/, optionally starts from init_weights.bin, optionally
#   limits to N samples (see the header comment of the .c).
# eval_exact_host       -- evaluates the exact accuracy (host) of any
#   model.bin, including the one produced by training under AxRAM.
# gen_init_weights_host -- generates reproducible initial weights (same
#   seed -> same weights), so the exact baseline and training under AxRAM
#   start from the same point.
host: train_exact_host eval_exact_host gen_init_weights_host

train_exact_host: $(TRAIN_VAL_DIR)/train_exact_host.c $(LIB_SRC) $(LIB_HDR)
	$(CC) -O2 $(LIB_FLAGS) -o $@ $(TRAIN_VAL_DIR)/train_exact_host.c $(LIB_SRC) -lm

eval_exact_host: $(TRAIN_VAL_DIR)/eval_exact_host.c $(LIB_SRC) $(LIB_HDR)
	$(CC) -O2 $(LIB_FLAGS) -o $@ $(TRAIN_VAL_DIR)/eval_exact_host.c $(LIB_SRC) -lm

gen_init_weights_host: $(TRAIN_AXRAM_DIR)/gen_init_weights_host.c $(LIB_SRC) $(LIB_HDR)
	$(CC) -O2 $(LIB_FLAGS) -o $@ $(TRAIN_AXRAM_DIR)/gen_init_weights_host.c $(LIB_SRC) -lm

# validate_kernel_axpike -- VALIDATION/INFERENCE kernel under AxPike (one
#   forward pass per invocation).
# train_kernel_axpike    -- TRAINING kernel under AxPike (the whole
#   epochs/samples loop in a single invocation -- see the header comment
#   of the .cpp to understand why it has to be this way).
riscv: validate_kernel_axpike train_kernel_axpike

validate_kernel_axpike: $(TRAIN_VAL_DIR)/validate_kernel_axpike.cpp $(LIB_SRC) $(LIB_HDR)
	$(RV_CXX) $(RV_FLAGS) $(LIB_FLAGS) -o $@ $(TRAIN_VAL_DIR)/validate_kernel_axpike.cpp $(LIB_SRC) -lm

train_kernel_axpike: $(TRAIN_AXRAM_DIR)/train_kernel_axpike.cpp $(LIB_SRC) $(LIB_HDR)
	$(RV_CXX) $(RV_FLAGS) $(LIB_FLAGS) -o $@ $(TRAIN_AXRAM_DIR)/train_kernel_axpike.cpp $(LIB_SRC) -lm

# Optional: trains exactly on the host and generates model.bin (random
# initial weights, EPOCHS epochs). Not part of `make all` and not used by
# train_sweep.py, validate_matrix.py or genann_report.py.
train: dataset host
	./train_exact_host $(HIDDEN_UNITS) $(EPOCHS) $(HIDDEN_LAYERS) > model.bin

# Generates reproducible initial weights (same HIDDEN_UNITS/HIDDEN_LAYERS/
# INIT_SEED -> always the same weights) -- used both by the exact baseline
# and by training under AxRAM, so they start from the same point. Only makes
# sense to run manually before train_axram/run_train_axram.py:
#
#     make init_weights
#     python3 train_axram/run_train_axram.py
init_weights: host
	./gen_init_weights_host $(HIDDEN_UNITS) $(HIDDEN_LAYERS) $(INIT_SEED) > init_weights.bin

# Removes only the binaries/objects -- keeps dataset_digits/, model.bin and
# init_weights.bin.
clean:
	rm -f train_exact_host eval_exact_host gen_init_weights_host \
	      validate_kernel_axpike train_kernel_axpike *.o

# Removes everything this Makefile generates, including dataset, trained model,
# initial weights and the model trained under AxRAM.
distclean: clean
	rm -rf dataset_digits model.bin init_weights.bin model_trained_axram_err*.bin
