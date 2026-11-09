export function FilmMark() {
  return (
    <svg aria-hidden="true" width="23" height="23" viewBox="0 0 24 24" fill="none">
      <rect x="2.75" y="4.25" width="18.5" height="15.5" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <path d="M7 4.5v15m10-15v15M3 9h4m-4 6h4m10-6h4m-4 6h4" stroke="currentColor" strokeWidth="1.5" />
      <path d="m11 9 4 3-4 3z" fill="currentColor" />
    </svg>
  );
}

export function SearchIcon() {
  return (
    <svg aria-hidden="true" width="21" height="21" viewBox="0 0 24 24" fill="none">
      <circle cx="10.75" cy="10.75" r="5.75" stroke="currentColor" strokeWidth="1.7" />
      <path d="m15 15 4.25 4.25" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export function Star({filled = false}: {filled?: boolean}) {
  return (
    <svg aria-hidden="true" width="23" height="23" viewBox="0 0 24 24" fill={filled ? 'currentColor' : 'none'}>
      <path d="m12 2.8 2.74 5.56 6.14.9-4.44 4.32 1.05 6.11L12 16.8l-5.49 2.89 1.05-6.11-4.44-4.32 6.14-.9L12 2.8Z" stroke="currentColor" strokeWidth="1.55" strokeLinejoin="round" />
    </svg>
  );
}
