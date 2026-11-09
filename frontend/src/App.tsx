import {lazy, Suspense, useEffect, useRef, useState} from 'react';
import {Discovery} from './Discovery';
import {FilmMark} from './Icons';

const ModelLab = lazy(() => import('./ModelLab'));

const PROJECT = 'https://github.com/NikosMav/movielens-recommender';

export function App() {
  const [lab, setLab] = useState(() => location.hash === '#model-lab');
  const main = useRef<HTMLElement>(null);

  useEffect(() => {
    function onNavigation() {
      setLab(location.hash === '#model-lab');
      window.scrollTo(0, 0);
      main.current?.focus({preventScroll: true});
    }
    window.addEventListener('hashchange', onNavigation);
    return () => window.removeEventListener('hashchange', onNavigation);
  }, []);

  return (
    <>
      <a className="skip-link" href="#main-content" onClick={(event) => {
        event.preventDefault();
        document.getElementById('main-content')?.focus();
      }}>Skip to content</a>
      <header className="site-header">
        <div className="page-width header-content">
          <a className="brand" href="#discover">
            <span><FilmMark /></span> MovieLens
          </a>
          <nav aria-label="Main navigation">
            <a href="#discover" aria-current={lab ? undefined : 'page'}>Discover</a>
            <a href="#model-lab" aria-current={lab ? 'page' : undefined}>Model Lab</a>
          </nav>
        </div>
      </header>
      <main id="main-content" className="page-width" tabIndex={-1} ref={main}>
        <Discovery active={!lab} />
        {lab && (
          <Suspense fallback={<p className="model-lab" role="status">Opening Model Lab…</p>}>
            <ModelLab />
          </Suspense>
        )}
      </main>
      <footer className="site-footer page-width">
        <p>Recommendations use the MovieLens 1M dataset from GroupLens.</p>
        <div>
          <a href="https://grouplens.org/datasets/movielens/1m/">About the dataset</a>
          <a href={PROJECT + '#results'}>Model evaluation</a>
          <a href={PROJECT}>GitHub</a>
        </div>
      </footer>
    </>
  );
}
