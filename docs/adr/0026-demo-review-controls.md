# ADR-0026: Truthful discovery controls and a usable phone flow

## Status

Accepted.

## Context

The [November demo review](../reviews/2026-11-11-demo-review.md) found that
sharing, changed ratings, slider-adjusted lists, and low-star explanations could
disagree with what the page claimed. The phone ratings action followed the
whole browsing shelf. This decision refines ADR-0024 and supersedes the sharing,
sample-profile, and stale-pick behaviors described in ADR-0025.

## Decision

- Sharing reveals a canonical public URL with movie IDs, stars, and slider
  position. The native code-box copy control performs copying. No clipboard
  success is claimed by the server. No identity or demographic fields are added.
- A fresh shared-link session restores both the profile and slider position.
- The missing-film answer identifies membership in the displayed list before
  reporting position in the measured candidate order. Search matches can be
  selected explicitly.
- A changed profile hides the previous picks and their tools until refresh.
- Liking in display copy requires four or five stars. Other history evidence is
  described as similarity to a rated film.
- Genre samples explicitly replace the current profile with up to four
  five-star ratings. They are labeled samples, not inferred preferences.
- The ratings summary appears before browsing on phones. Narrow shelves use
  compact rows so five 44px rating targets fit. Desktop keeps its browsing and
  summary columns. Missing-poster evidence stays thumbnail-sized.
- Section changes reset the main scroll position after the new view renders.

## Consequences

The measured default ranking and saved evaluation results do not change.
The adventurous control remains an unevaluated re-ranking; four fixed-profile
behavioral probes demonstrate its effect without establishing preference or
relevance improvements. No new dependency, model, account system, or database
is introduced. The archived Astryx client remains available for the local API.
