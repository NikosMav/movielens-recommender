.PHONY: space-snapshot demo api test lint

demo:
	streamlit run space/app.py

api:
	uvicorn --factory movielens_recommender.serving.api:create_app_from_env --port 8000

test:
	MOVIELENS_ALLOW_DOWNLOAD=0 pytest -q

lint:
	ruff check src tests scripts app space

space-snapshot:
	python scripts/build_space_snapshot.py
