env:
    conda create -n item_based_recommendations_with_llms python=3.10.12

install-deps:
    python -m pip install -r requirements.txt
    conda install -c conda-forge faiss-cpu

test:
    pytest --log-cli-level=DEBUG --cov=. --cov-report=html

lint:
    isort .
    black .