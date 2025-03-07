docker build --platform linux/amd64 --build-arg BASE_NAME=amd --build-arg VLLM_TARGET_DEVICE=rocm -t gdiamos/rocm-base:latest .
