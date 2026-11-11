# ADR-0027: Readable benchmark comparisons and pipeline diagram

## Status

Accepted.

## Decision

The quality page starts with a focused comparison of the full model,
item similarity, two-tower retrieval, and popularity. The highest stored
scoring model is always included, even if a future model wins outside that
set. All recorded variants remain in an expandable comparison, and model
IDs and exact values remain in Detailed scores.

Charts start at zero, use a labelled scale with headroom, and distinguish
score dots from thin uncertainty whiskers. Their geometry uses the stored
floating-point values; display labels round to three decimal places.
The highest score is labelled as a stored point, without treating that label
as a significance test. Overlapping ranges are described as overlapping.

The pipeline diagram explains the four stages and uses live counts only when
a request supplied them. Without a request it shows explanations rather than
empty count placeholders. A small set of decorative SVG icons gives the
diagram and the main task headings a consistent visual cue, without adding
a runtime dependency or replacing their accessible text labels.

## Validation

Tests preserve the selected records, include an unexpected future leader,
check common point/interval geometry, and prevent invented diagram counts.
Stored results, trained models, default rankings, and the backend pin are
unchanged. Desktop, phone, light, and dark layouts are checked in the browser.
