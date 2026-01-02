################################################################################
# Makefile para compilar nvdsinfer_testparser_tao.cpp a libnvds_infercustomparser_test.so
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

DS_VER = $(shell deepstream-app -v | awk '$$1~/DeepStreamSDK/ {print substr($$2,1,3)}' )

DS_SRC_PATH := /opt/nvidia/deepstream/deepstream-$(DS_VER)
CC:= g++

# Base CFLAGS
CFLAGS+= -Wall -std=c++11 -shared -fPIC

# Deepstream SDK includes y CUDA includes
CFLAGS+= -I$(DS_SRC_PATH)/sources/includes \
	  -I/usr/local/cuda-$(CUDA_VER)/include

# Debug logger header (si lo usas)
CFLAGS+= -I.

# Libraries
LIBS+= -lnvinfer -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

# Source files y targets
SRCFILES:= nvdsinfer_testparser_tao.cpp
# Si tienes otros helpers, agregarlos aquí
# SRCFILES+= debug_logger_raii.cpp debug_logger_tensor.hpp

TARGET_LIB:= libnvds_infercustomparser_test.so

# Release build flags
CFLAGS_RELEASE = $(CFLAGS) -O2

# Debug build flags
CFLAGS_DEBUG = $(CFLAGS) -g -DDEBUG -O0

# Default target
all: $(TARGET_LIB)

# Release build
$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS_RELEASE) $(LFLAGS)

# Debug target
debug: $(TARGET_LIB)
	@echo "Debug build done: $(TARGET_LIB)"

clean:
	rm -f $(TARGET_LIB)

.PHONY: all debug clean
