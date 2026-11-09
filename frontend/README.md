# MovieLens frontend

A React 19 and Astryx interface for the local MovieLens serving API. Discover
uses the new-user model and keeps ratings in page memory. Model Lab uses an
existing profile from the production snapshot to show the recent training
ratings and three ranked lists.

## Run locally

Start FastAPI from the repository root with the Python API dependencies and
both serving snapshots available. See [the serving setup](../README.md#serve-over-http-batch-and-docker)
for snapshot commands and environment variables.

```sh
uvicorn --factory movielens_recommender.serving.api:create_app_from_env --port 8000
```

In a second terminal, run the frontend with Node.js 22.12 or later.

```sh
cd frontend
npm ci
npm run dev
```

Open the localhost URL printed by Vite. Set `MOVIELENS_API_URL` in the
frontend process if the API runs on a different port.

```sh
npm run build
npm run preview
```

The build writes static files to `dist/`. The browser sends requests to
`/v1` on the frontend origin. Vite proxies those requests in development
and preview. Any other host serving the build needs an equivalent proxy to the
local FastAPI service. The API is intended for local use.
