# Hugging Face Spaces Docker Image for Wolfgang-LM
FROM ghcr.io/prefix-dev/pixi:0.39.5

# HF Spaces requires the app to run as user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy pixi configuration first (better layer caching)
COPY --chown=user:user pyproject.toml pixi.lock ./

# Copy application code needed for installation (setup of python package)
COPY --chown=user:user wolfgang_lm/ ./wolfgang_lm/

# Install dependencies via pixi
RUN pixi install --locked

# Copy remaining assets (frontend, entrypoint)
COPY --chown=user:user web/ ./web/
COPY --chown=user:user entrypoint.sh .

# Make entrypoint executable and create directories with correct permissions
RUN chmod +x entrypoint.sh && \
    mkdir -p out-finetune && \
    mkdir -p hf_cache && \
    mkdir -p data_clean && \
    chown -R user:user out-finetune hf_cache data_clean

# Set HF Cache to local directory to avoid permission issues
ENV HF_HOME=/app/hf_cache

# Switch to non-root user
USER user

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Run the entrypoint script
CMD ["./entrypoint.sh"]
