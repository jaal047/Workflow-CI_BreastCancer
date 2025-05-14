# Gunakan base image
FROM continuumio/miniconda3

# Set Workdir
WORKDIR /app

# Copy file MLProject
COPY . .

# Install dependencies
RUN conda env create -f conda.yaml

# Aktifkan environment
SHELL ["conda", "run", "-n", "breast_cancer_env", "/bin/bash", "-c"]

# Jalankan script
CMD ["python", "modelling.py"]