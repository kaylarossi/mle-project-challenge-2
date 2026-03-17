# For more information, please refer to https://aka.ms/vscode-docker-python
FROM continuumio/miniconda3

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install pip requirements
COPY conda_environment.yml .
RUN conda env create -f conda_environment.yml && conda clean -a

COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["conda", "run", "--no-capture-output", "-n", "housing", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
