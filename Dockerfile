# Use the official ContinuumIO Miniconda3 image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy environment file
COPY conda_environment.yml ./

# Create the environment:
RUN conda env create -f conda_environment.yml

# Make RUN commands use the new environment:
SHELL ["/bin/bash", "-c"]

# Activate the environment and ensure it's on PATH:
ENV PATH /opt/conda/envs/$(head -1 conda_environment.yml | cut -d' ' -f2)/bin:$PATH

# Copy the rest of the project files
COPY . .

# Set the default command to activate the conda environment and run the script
CMD ["/bin/bash", "-c", "source activate housing && python create_model.py"]
#CMD ["python", "create_model.py"]