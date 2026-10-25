FROM python:3.11-slim

LABEL maintainer="Jan Wildenhain <jan.wildenhain@gmail.com>"
LABEL description="Modernized phpIoRN / GRN-Sim Gene Network Perturbation Simulator & Reverse-Engineering Engine (Wildenhain & Crampin 2006)"

# Install Graphviz (dot, neato) for hierarchical & spring layout SVG generation matching draw_dot.m / statsgraph_to_dot.m
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md SPECIFICATION.md ./
COPY data/ ./data/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install Python package and test suite dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[dev]"

# Run unit and benchmark verification tests during build
RUN pytest -v --ignore-glob="*playwright*"

ENV HOST=0.0.0.0
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

CMD ["python", "-m", "grn_sim.web.server", "8080"]
