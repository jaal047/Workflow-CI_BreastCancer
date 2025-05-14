# Gunakan base image Miniconda
FROM continuumio/miniconda3

# Set Workdir
WORKDIR /app

# Copy semua file ke dalam Docker container
COPY . .

# Install dependencies
RUN conda env create -f conda.yaml

# Aktifkan environment secara manual dan install dependencies tambahan (jika ada)
RUN echo "conda activate breast_cancer_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Pastikan dependencies terpasang
RUN conda run -n breast_cancer_env pip install mlflow dagshub joblib

# Expose port untuk serving
EXPOSE 8000

# Jalankan script dengan environment yang diaktifkan
CMD ["conda", "run", "--no-capture-output", "-n", "breast_cancer_env", "python", "modelling.py"]
