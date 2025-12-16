CXX      := g++
CXXFLAGS ?= -std=c++17 -O2 -Wall

KERNEL_DIR          	 ?= kernels
TEST_DIR            	 ?= tests
ALPAKA_DIR          	 ?= $(CURDIR)/external/alpaka/include
CPLUS_INCLUDE_PATH       ?= /opt/homebrew/include
BIN_DIR                  ?= bin
ALPAKA_ACCELERATOR_FLAG  ?= ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

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
	$(CXX) $(CXXFLAGS) -I$(ALPAKA_DIR) -I$(CPLUS_INCLUDE_PATH) -D$(ALPAKA_ACCELERATOR_FLAG) $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

test:

.PHONY = all test clean
