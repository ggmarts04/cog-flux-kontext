# Base image from RunPod with PyTorch, Python 3.11, and CUDA 12.1.1
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install pget for faster downloads (from cog.yaml)
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && \
    chmod +x /usr/local/bin/pget

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Purge pip cache and print requirements.txt content for debugging
RUN pip cache purge
RUN echo "--- Contents of requirements.txt ---" && \
    cat requirements.txt && \
    echo "------------------------------------"

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Using --extra-index-url for PyTorch if needed, but torch is pinned in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the working directory
COPY . .

# Apply the torch patch (from cog.yaml)
# Note: The original path uses /root/.pyenv/versions/3.11.12/...
# We need to find the correct site-packages path in this Docker image.
# A common path is /usr/local/lib/python3.11/site-packages/
# We'll try that, but this might need adjustment if the image structure is different.
# Listing the directory contents during image build or inspecting the image would confirm this.
# For now, we'll assume a standard path.
RUN PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    echo "Python site-packages found at: $PYTHON_SITE_PACKAGES" && \
    wget -O $PYTHON_SITE_PACKAGES/torch/_inductor/fx_passes/post_grad.py \
    https://gist.githubusercontent.com/alexarmbr/d3f11394d2cb79300d7cf2a0399c2605/raw/378fe432502da29f0f35204b8cd541d854153d23/patched_torch_post_grad.py

# Ensure the models directory exists and is writable if weights are downloaded there
RUN mkdir -p /app/models && chmod -R 777 /app/models

# Expose the port RunPod expects (though for serverless, this is less critical)
# EXPOSE 8080

# Command to run the RunPod handler
CMD ["python", "-u", "rp_handler.py"]
