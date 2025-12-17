CXX      := g++
CXXFLAGS ?= -std=c++17 -O3 -Wall
LDFLAGS  ?=

# Path setup: Mac + Linux compatible
KERNEL_DIR          	 ?= kernels
TEST_DIR            	 ?= tests
ALPAKA_DIR          	 ?= $(CURDIR)/external/alpaka/include
CPLUS_INCLUDE_PATH       ?= /opt/homebrew/include
LIBRARY_PATH         	 ?= /opt/homebrew/lib
BIN_DIR                  ?= bin

LDFLAGS += -L$(LIBRARY_PATH)

# Accelerator selection (CPU options)
# Debugging (slow, checks everything)
# ALPAKA_ACCELERATOR_FLAG  ?= ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

# Performance (fast, single core)
# ALPAKA_ACCELERATOR_FLAG ?= ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

# High performance (fast, multi-core TBB)
ALPAKA_ACCELERATOR_FLAG ?= ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

# Conditional flags
# Auto-add -ltbb if TBB is selected
ifneq (,$(findstring TBB,$(ALPAKA_ACCELERATOR_FLAG)))
    LDFLAGS += -ltbb
endif

# Auto-add -fopenmp if OMP is selected
ifneq (,$(findstring OMP,$(ALPAKA_ACCELERATOR_FLAG)))
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
endif

# Build rules
KERNEL_HEADERS := $(wildcard $(KERNEL_DIR)/*.hpp)
KERNEL_NAMES := $(patsubst $(KERNEL_DIR)/%.hpp,%,$(KERNEL_HEADERS))
EXECUTABLES := $(patsubst %,$(BIN_DIR)/test_%.out,$(KERNEL_NAMES))

all: $(EXECUTABLES)

test: $(EXECUTABLES)
	@echo "Starting execution of all tests..."
	@for exe in $(EXECUTABLES); do \
		echo "----------------------------------------"; \
		echo "Running $$exe"; \
		./$$exe || exit 1; \
	done
	@echo "----------------------------------------"
	@echo "All tests passed successfully!"

$(BIN_DIR)/test_%.out: $(TEST_DIR)/test_%.cpp $(KERNEL_DIR)/%.hpp | $(BIN_DIR)
	@echo "Building test for kernel: $*"
	$(CXX) $(CXXFLAGS) -I$(ALPAKA_DIR) -I$(CPLUS_INCLUDE_PATH) -D$(ALPAKA_ACCELERATOR_FLAG) $< -o $@ $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all test clean
