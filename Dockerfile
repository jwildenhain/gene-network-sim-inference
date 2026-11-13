FROM python:3.11-slim

LABEL maintainer="Jan Wildenhain <jan.wildenhain@gmail.com>"
LABEL description="Modernized phpIoRN / GRN-Sim Gene Network Perturbation Simulator & Reverse-Engineering Engine (Wildenhain & Crampin 2006)"

# Graphviz (dot, neato) renders the hierarchical and spring layouts; the studio falls back to a built-in layout without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# One BLAS thread, set before anything loads NumPy (the tests below included): OpenBLAS's thread pool and fork() deadlock the process pools (see src/grn_sim/__init__.py)
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install the package the way any user would (the wheel carries the web page and the bundled data), not in editable mode.
COPY pyproject.toml README.md SPECIFICATION.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[dev,numba]"

# Run the unit and benchmark tests during the build, against the installed package (the default). On a shared machine build with
#   docker compose build --build-arg RUN_TESTS=0
# and run the same suite pinned to some cores afterwards:  docker run --cpuset-cpus 0-7 --entrypoint "" <image> pytest -p no:cacheprovider --ignore-glob="*playwright*"
ARG RUN_TESTS=1
COPY scripts/ ./scripts/
COPY tests/ ./tests/
RUN if [ "$RUN_TESTS" = "1" ]; then pytest -v -p no:cacheprovider --ignore-glob="*playwright*"; else echo "tests skipped in the build (RUN_TESTS=$RUN_TESTS)"; fi

# Everything the studio writes (simulation archive, exports, downloads) lives in /app/data: mount a volume there
ENV GRN_SIM_HOME=/app/data
ENV HOST=0.0.0.0
ENV PORT=8080
VOLUME ["/app/data"]
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

CMD ["python", "-m", "grn_sim.web.server", "8080"]
