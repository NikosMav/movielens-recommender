import type {Recommendation} from './api';

export function Genres({value}: {value: string}) {
  const genres = value.split('|').filter((genre) => genre && genre !== '(no genres listed)');
  return genres.length ? <span>{genres.join(', ')}</span> : null;
}

export function Recommendations({movies, titleLevel: Title = 'h3'}: {movies: Recommendation[]; titleLevel?: 'h3' | 'h4'}) {
  if (!movies.length) return <p>The model did not return any titles for these ratings.</p>;

  return (
    <ol className="recommendations" role="list">
      {movies.map((movie, index) => {
        const reasons = [...new Set(movie.reasons ?? [])];
        return (
          <li key={movie.item_id}>
            <span className="movie-rank" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
            <div className="movie-copy">
              <Title className="movie-title"><span className="sr-only">Rank {index + 1}. </span>{movie.title}</Title>
              <div className="movie-meta">{movie.year && <span>{movie.year}</span>}<Genres value={movie.genres} /></div>
              {reasons[0] && <p>{reasons[0]}</p>}
              {reasons.length > 1 && (
                <details>
                  <summary>Why this pick? <span className="sr-only">More about {movie.title}</span></summary>
                  <ul>{reasons.slice(1).map((reason) => <li key={reason}>{reason}</li>)}</ul>
                </details>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
