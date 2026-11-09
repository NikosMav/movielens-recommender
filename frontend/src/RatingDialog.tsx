import {useState} from 'react';
import {Button} from '@astryxdesign/core/Button';
import {Dialog, DialogHeader} from '@astryxdesign/core/Dialog';
import type {Movie} from './api';
import {Star} from './Icons';

export default function RatingDialog({movie, initialRating, onClose, onSave}: {
  movie: Movie;
  initialRating?: number;
  onClose: () => void;
  onSave: (movie: Movie, rating: number) => void;
}) {
  const [rating, setRating] = useState(initialRating ?? 0);

  return (
    <Dialog isOpen onOpenChange={(open) => {if (!open) onClose();}} width={440}>
      <DialogHeader title={movie.title} subtitle={movie.year} onOpenChange={onClose} />
      <form className="rating-dialog" onSubmit={(event) => {
        event.preventDefault();
        if (rating > 0) onSave(movie, rating);
      }}>
        <fieldset className="star-picker">
          <legend>How did you like it?</legend>
          <div className="star-options">
            {[1, 2, 3, 4, 5].map((stars) => (
              <label key={stars} className={stars <= rating ? 'star-filled' : ''}>
                <input
                  type="radio"
                  name="rating"
                  value={stars}
                  checked={rating === stars}
                  onChange={() => setRating(stars)}
                  aria-label={stars + (stars === 1 ? ' star' : ' stars')}
                />
                <Star filled={stars <= rating} />
                <span aria-hidden="true">{stars}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <Button className="main-button" type="submit" variant="primary" size="lg"
          label={initialRating ? 'Update rating' : 'Save rating'} isDisabled={!rating} width="100%" />
      </form>
    </Dialog>
  );
}
