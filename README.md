# End-to-End MLOps Pipeline

## Flow

GitHub code/data change
-> GitHub Actions
-> data validation
-> tests
-> model training
-> MLflow tracking
-> model quality gate
-> Docker build
-> Docker Hub
-> EC2 deployment
-> FastAPI prediction API

## Project structure

```text
mlops-house-price/
├── data/housing.csv
├── src/
│   ├── validate_data.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── app/main.py
├── model/model.pkl
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── .github/workflows/mlops.yml
```

## Local commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.validate_data
python -m src.train
pytest -q

mlflow ui --backend-store-uri ./mlruns

uvicorn app.main:app --reload
```

Open FastAPI docs at `http://127.0.0.1:8000/docs`.

## Docker

```bash
docker build -t mlops-house-price .
docker run -d --name mlops-house-price -p 8000:8000 mlops-house-price
```

## Docker Hub

```bash
docker login
docker tag mlops-house-price YOUR_DOCKERHUB_USERNAME/mlops-house-price:latest
docker push YOUR_DOCKERHUB_USERNAME/mlops-house-price:latest
```

## EC2

Amazon Linux:

```bash
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
```

Reconnect after changing the group.

Security Group:
- TCP 22: SSH, preferably your IP only
- TCP 80: website/API

Manual test:

```bash
sudo docker pull YOUR_DOCKERHUB_USERNAME/mlops-house-price:latest

sudo docker run -d   --name mlops-house-price   -p 80:8000   YOUR_DOCKERHUB_USERNAME/mlops-house-price:latest
```

Open `http://EC2_PUBLIC_IP/docs`.

## GitHub Secrets

Create:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
EC2_HOST
EC2_USERNAME
EC2_SSH_KEY
```

For Amazon Linux:

```text
EC2_USERNAME = ec2-user
```

Paste the complete `.pem` into `EC2_SSH_KEY`, including:

```text
-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----
```

## What triggers the pipeline?

The workflow triggers on changes to:

```text
data/**
src/**
tests/**
app/**
Dockerfile
requirements.txt
.github/workflows/mlops.yml
```

So both a new dataset and a training-code change trigger the pipeline.

## Model quality gate

`R2_THRESHOLD = 0.85`.

If R2 is below 0.85, the pipeline fails before Docker build/deployment.

## Classroom demonstration

1. Show the model API running on EC2.
2. Change `data/housing.csv`.
3. `git add`, commit and push.
4. Show GitHub Actions.
5. Show validation.
6. Show tests.
7. Show MLflow metrics.
8. Show Docker build and Docker Hub.
9. Show EC2 deployment.
10. Refresh `/docs`.
11. Change `src/train.py`.
12. Push again and show the same MLOps pipeline.

## DevOps vs MLOps

DevOps:
Code -> Test -> Build -> Docker -> Deploy

MLOps:
Code + Data -> Validate -> Test -> Train -> MLflow -> Evaluate -> Quality Gate -> Docker -> Registry -> Deploy

### Note

The GitHub runner is temporary. The workflow uploads `mlruns/` as a GitHub Actions artifact so the MLflow run data is retained for the demonstration. Production systems should use a persistent MLflow Tracking Server and persistent artifact storage.
