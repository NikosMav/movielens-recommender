import {lazy, Suspense, useEffect, useMemo, useRef, useState} from 'react';
import {Banner} from '@astryxdesign/core/Banner';
import {Button} from '@astryxdesign/core/Button';
import {Typeahead, type SearchSource} from '@astryxdesign/core/Typeahead';
import {getRecommendations, searchMovies, type Movie, type Rating, type Recommendation} from './api';
import {FilmMark, SearchIcon} from './Icons';
import {Recommendations} from './Recommendations';

const RatingDialog = lazy(() => import('./RatingDialog'));

const MIN_RATINGS = 3;
const MAX_RATINGS = 200;
const STARTERS = ['The Matrix', 'Toy Story', 'Fargo'];

type MovieOption = {id: string; label: string; movie: Movie};

export function Discovery({active}: {active: boolean}) {
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [editing, setEditing] = useState<Movie | null>(null);
  const [starter, setStarter] = useState<string | null>(null);
  const [searchError, setSearchError] = useState('');
  const [result, setResult] = useState<Recommendation[] | null>(null);
  const [resultError, setResultError] = useState('');
  const [pending, setPending] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const recommendationRequest = useRef<AbortController | null>(null);
  const starterRequest = useRef<AbortController | null>(null);
  const remaining = Math.max(MIN_RATINGS - ratings.length, 0);

  const source = useMemo<SearchSource<MovieOption>>(() => {
    let request: AbortController | undefined;
    return {
      bootstrap: () => [],
      cancel: () => request?.abort(),
      async search(query) {
        const title = query.trim();
        if (title.length < 2) return [];
        request?.abort();
        const current = new AbortController();
        request = current;
        setSearchError('');
        try {
          const {movies} = await searchMovies(title, current.signal);
          return movies.map((movie) => ({
            id: String(movie.item_id),
            label: movie.year ? movie.title + ' (' + movie.year + ')' : movie.title,
            movie,
          }));
        } catch {
          if (!current.signal.aborted) {
            setSearchError('Could not search the catalog. Check that the model service is running and try again.');
          }
          return [];
        }
      },
    };
  }, []);

  useEffect(() => {
    if (result) resultHeading.current?.focus();
  }, [result]);

  useEffect(() => {
    if (!active) {
      starterRequest.current?.abort();
      setStarter(null);
    }
  }, [active]);

  function openRating(movie: Movie) {
    if (ratings.length >= MAX_RATINGS && !ratings.some((row) => row.item_id === movie.item_id)) {
      setSearchError('This profile already has 200 ratings. Remove a rating before adding another.');
      return;
    }
    setEditing(movie);
  }

  async function pickStarter(title: string) {
    starterRequest.current?.abort();
    const request = new AbortController();
    starterRequest.current = request;
    setStarter(title);
    setSearchError('');
    try {
      const {movies} = await searchMovies(title, request.signal);
      if (request.signal.aborted) return;
      const movie = movies.find((row) => row.title.toLowerCase() === title.toLowerCase());
      if (movie) openRating(movie);
      else setSearchError(title + ' is unavailable in this catalog. Try searching for another film.');
    } catch {
      if (!request.signal.aborted) setSearchError('Could not load that title. Try again or search for a film.');
    } finally {
      if (!request.signal.aborted) setStarter(null);
    }
  }

  function clearResult() {
    recommendationRequest.current?.abort();
    setPending(false);
    setResult(null);
    setResultError('');
  }

  function saveRating(movie: Movie, rating: number) {
    setRatings((current) => {
      const entry = {...movie, rating};
      return current.some((row) => row.item_id === movie.item_id)
        ? current.map((row) => row.item_id === movie.item_id ? entry : row)
        : [...current, entry];
    });
    clearResult();
    setEditing(null);
  }

  async function recommend() {
    if (remaining || pending) return;
    const request = new AbortController();
    recommendationRequest.current = request;
    setPending(true);
    setResultError('');
    try {
      const data = await getRecommendations(ratings, request.signal);
      if (!request.signal.aborted) setResult(data.recommendations);
    } catch {
      if (!request.signal.aborted) setResultError('Could not get your picks. Your ratings are still here. Try again.');
    } finally {
      if (!request.signal.aborted) setPending(false);
    }
  }

  return (
    <div hidden={!active}>
      <section className="intro" aria-labelledby="discover-title">
        <div className="intro-copy">
          <h1 id="discover-title">Find your<br /> next film.</h1>
          <p>Rate three films you’ve seen. MovieLens will find ten more to try and tell you why they made the list.</p>
          <Button className="main-button start-button" variant="primary" size="lg"
            label="Rate a film" onClick={() => input.current?.focus()} />
        </div>
        <div className="intro-art" aria-hidden="true">
          <img src={import.meta.env.BASE_URL + 'screening-room.svg'} width="700" height="424" alt="" />
        </div>
      </section>

      <div className="workbench">
        <section className="picker" aria-labelledby="picker-title">
          <h2 id="picker-title">What have you watched?</h2>
          <p>Search for a title, then give it a rating from one to five stars.</p>
          <Typeahead
            ref={input}
            label="Search for a movie"
            isLabelHidden
            placeholder="Search for a movie"
            startIcon={<SearchIcon />}
            containerClassName="movie-search"
            controlClassName="movie-search-control"
            width="100%"
            size="lg"
            minQueryLength={2}
            searchSource={source}
            value={null}
            onChange={(option) => {
              if (option) {
                starterRequest.current?.abort();
                setStarter(null);
                openRating(option.movie);
              }
            }}
            onChangeQuery={() => setSearchError('')}
            emptySearchText={searchError ? '' : 'No titles found. Try a shorter title or another film.'}
          />
          <div className="starters">
            <span>Try a title</span>
            {STARTERS.map((title) => (
              <Button key={title} className="starter-button" label={title} variant="secondary"
                isLoading={starter === title} onClick={() => void pickStarter(title)} />
            ))}
          </div>
          {searchError && <Banner status="error" title={searchError} role="alert" />}
        </section>

        <section className="rating-ticket" aria-labelledby="ratings-title">
          <div className="ticket-heading">
            <span className="ticket-mark"><FilmMark /></span>
            <h2 id="ratings-title">Your ratings</h2>
            <span className="ticket-count" aria-live="polite">{ratings.length} rated</span>
          </div>
          <div className="progress" aria-hidden="true">
            {[1, 2, 3].map((step) => <span key={step} className={step <= ratings.length ? 'rated' : ''} />)}
          </div>
          <p className="ticket-note" aria-live="polite">
            {remaining ? 'Rate ' + remaining + ' more ' + (remaining === 1 ? 'film' : 'films') + ' to get your picks.' : 'You can get picks now or rate a few more.'}
          </p>
          {!!ratings.length && (
            <ul className="ratings-list" role="list">
              {ratings.map((row) => (
                <li key={row.item_id}>
                  <button className="edit-rating" type="button"
                    aria-label={'Edit rating for ' + row.title + ', ' + row.rating + ' out of 5'}
                    onClick={() => openRating(row)}>
                    <span>{row.title}</span><span className="rated-stars">{row.rating} ★</span>
                  </button>
                  <button className="remove-rating" type="button" aria-label={'Remove ' + row.title}
                    onClick={() => {
                      setRatings((current) => current.filter((item) => item.item_id !== row.item_id));
                      clearResult();
                      input.current?.focus();
                    }}>×</button>
                </li>
              ))}
            </ul>
          )}
          <Button className="main-button" label="Get my picks" variant="primary" size="lg" width="100%"
            isDisabled={remaining > 0} isLoading={pending} onClick={() => void recommend()} />
          {resultError && <Banner status="error" title={resultError} role="alert" />}
        </section>
      </div>

      {result && (
        <section className="results" aria-labelledby="picks-title">
          <h2 id="picks-title" ref={resultHeading} tabIndex={-1}>Your next watch</h2>
          <p>Ranked using the films you rated.</p>
          <Recommendations movies={result} />
        </section>
      )}
      {editing && active && (
        <Suspense fallback={<p className="sr-only" role="status">Opening rating form…</p>}>
          <RatingDialog key={editing.item_id} movie={editing}
            initialRating={ratings.find((row) => row.item_id === editing.item_id)?.rating}
            onClose={() => setEditing(null)} onSave={saveRating} />
        </Suspense>
      )}
    </div>
  );
}
