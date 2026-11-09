export type Movie = {
  item_id: number;
  title: string;
  year: number | null;
};

export type Recommendation = Movie & {genres: string; reasons?: string[]};

export type Rating = Movie & {rating: number};

export type ModelComparison = {
  user_id: number;
  n_ratings: number;
  top_genres: string[];
  recent_ratings: {title: string; year: number | ''; genres: string; stars: string}[];
  recommendations: {
    production: Recommendation[];
    item_item: Recommendation[];
    two_tower: Recommendation[];
  };
};

class ApiError extends Error {
  constructor(readonly status: number) {
    super('Request failed (' + status + ')');
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch('/v1' + path, options);
  if (!response.ok) throw new ApiError(response.status);
  return response.json() as Promise<T>;
}

export function searchMovies(query: string, signal?: AbortSignal) {
  const params = new URLSearchParams({q: query, limit: '10'});
  return request<{movies: Movie[]}>('/movies/search?' + params, {signal});
}

export function getRecommendations(ratings: Rating[], signal: AbortSignal) {
  return request<{recommendations: Recommendation[]}>('/recommendations/new-user', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      ratings: ratings.map(({item_id, rating}) => ({item_id, rating})),
      n: 10,
    }),
    signal,
  });
}

export function getModelComparison(id: number, signal: AbortSignal) {
  return request<ModelComparison>('/users/' + id + '/comparison', {signal});
}

export function modelError(error: unknown) {
  if (error instanceof ApiError && error.status === 404) {
    return 'That profile is not in this MovieLens snapshot. Try profile 155.';
  }
  return 'Could not load this profile. Check that the model service is running and try again.';
}
