# Serving image. The runtime needs the standard library only, so the image is
# python-slim with the package and its artefacts and nothing else. Training
# dependencies are deliberately absent: the training stack is an order of
# magnitude larger than the service and has no business in a trading-path image.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.txt ./
COPY src ./src
RUN pip install --no-cache-dir .

# Lexicons, thresholds and the primary-source index are part of the release:
# a flag must be traceable to the exact lexicon and config that raised it.
COPY configs ./configs
COPY artifacts ./artifacts

# Run unprivileged. The service writes only to the decision log.
RUN useradd --create-home --uid 10001 fintext \
    && mkdir -p /app/reports \
    && chown -R fintext:fintext /app
USER fintext

EXPOSE 8080
ENV FINTEXT_HOST='"0.0.0.0"'

# The health check uses the stdlib server's own endpoint rather than curl, so
# the image needs no extra packages.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "-m", "fintext.cli"]
CMD ["serve"]
