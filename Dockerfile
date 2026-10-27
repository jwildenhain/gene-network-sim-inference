FROM python:3.11-slim

LABEL maintainer="Jan Wildenhain <jan.wildenhain@gmail.com>"
LABEL description="Modernized phpIoRN / GRN-Sim Gene Network Perturbation Simulator & Reverse-Engineering Engine (Wildenhain & Crampin 2006)"

# Graphviz (dot, neato) renders the hierarchical and spring layouts; the studio falls back to a built-in layout without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package the way any user would (the wheel carries the web page and the bundled data), not in editable mode.
COPY pyproject.toml README.md SPECIFICATION.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[dev]"

# Run the unit and benchmark tests during the build, against the installed package
COPY scripts/ ./scripts/
COPY tests/ ./tests/
RUN pytest -v -p no:cacheprovider --ignore-glob="*playwright*"

# Everything the studio writes (simulation archive, exports, downloads) lives in /app/data: mount a volume there
ENV GRN_SIM_HOME=/app/data
ENV HOST=0.0.0.0
ENV PORT=8080
VOLUME ["/app/data"]
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

CMD ["python", "-m", "grn_sim.web.server", "8080"]
