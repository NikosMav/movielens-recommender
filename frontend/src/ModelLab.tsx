import {useEffect, useRef, useState} from 'react';
import {Banner} from '@astryxdesign/core/Banner';
import {Button} from '@astryxdesign/core/Button';
import {TextInput} from '@astryxdesign/core/TextInput';
import {getModelComparison, modelError, type ModelComparison} from './api';
import {Genres, Recommendations} from './Recommendations';

const MODELS = [
  {id: 'production', title: 'Production ranker', description: 'Candidates reordered by LambdaRank.'},
  {id: 'item_item', title: 'Item–item cosine', description: 'Ranks by similarity to films in the profile.'},
  {id: 'two_tower', title: 'Two-tower retrieval', description: 'Ranks by learned user and movie embeddings.'},
] as const;

export default function ModelLab() {
  const [profileId, setProfileId] = useState('155');
  const [profile, setProfile] = useState<ModelComparison | null>(null);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const request = useRef<AbortController | null>(null);
  const resultsHeading = useRef<HTMLHeadingElement>(null);

  useEffect(() => () => request.current?.abort(), []);

  useEffect(() => {
    if (profile) resultsHeading.current?.focus();
  }, [profile]);

  async function loadProfile() {
    const id = Number(profileId);
    if (!/^\d+$/.test(profileId.trim()) || !Number.isSafeInteger(id) || id < 1) {
      setError('Enter a positive whole number for the profile ID.');
      return;
    }
    request.current?.abort();
    const current = new AbortController();
    request.current = current;
    setError('');
    setProfile(null);
    setPending(true);
    try {
      const data = await getModelComparison(id, current.signal);
      if (!current.signal.aborted) setProfile(data);
    } catch (error) {
      if (!current.signal.aborted) setError(modelError(error));
    } finally {
      if (!current.signal.aborted) setPending(false);
    }
  }

  return (
    <section className="model-lab" aria-labelledby="lab-title">
      <div className="lab-intro">
        <h1 id="lab-title">Model Lab</h1>
        <p>Pick a MovieLens profile and compare what three models recommend from the same rating history.</p>
      </div>
      <form className="lab-form" onSubmit={(event) => {event.preventDefault(); void loadProfile();}}>
        <TextInput label="MovieLens profile ID" value={profileId} size="lg" inputMode="numeric"
          onChange={setProfileId} />
        <Button className="main-button" label="Load profile" type="submit" size="lg"
          variant="primary" isLoading={pending} />
      </form>
      {error && <Banner status="error" title={error} role="alert" />}
      {profile && (
        <div className="lab-results">
          <h2 ref={resultsHeading} tabIndex={-1}>Profile {profile.user_id}</h2>
          <p>{profile.n_ratings} training ratings{profile.top_genres.length > 0 && ' / ' + profile.top_genres.join(', ')}</p>
          <details className="history">
            <summary>Recent ratings</summary>
            <ul>
              {profile.recent_ratings.map((movie, index) => (
                <li key={index}>
                  <strong>{movie.title}{movie.year ? ' (' + movie.year + ')' : ''}</strong>
                  <Genres value={movie.genres} />
                  <span>{movie.stars}</span>
                </li>
              ))}
            </ul>
          </details>
          <div className="comparison">
            {MODELS.map((model) => (
              <section key={model.id} aria-labelledby={model.id}>
                <h3 id={model.id}>{model.title}</h3>
                <p>{model.description}</p>
                <Recommendations movies={profile.recommendations[model.id]} titleLevel="h4" />
              </section>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
