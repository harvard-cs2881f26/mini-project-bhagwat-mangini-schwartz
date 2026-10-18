# Presentation deck

`asimov-2.0-deck.html` is the team deck (35 slides). It's one self-contained file: fonts and images are embedded, so it opens offline.

**View it:** open the file in Chrome or Safari (double-click, or `open docs/deck/asimov-2.0-deck.html`). Scroll or use the arrow keys to move between slides. To make a PDF, print to PDF from the browser (each slide prints as one page). GitHub shows the source, not the slides.

## Editing it (you or your agent)

The file is about 4 MB, almost all of it embedded images. Don't read it whole. Find the slide you want, then make exact-text edits.

- **Each slide is one line**, a `<section class="deck-slide" id="N" ... data-label="Title">`. List them:
  ```bash
  grep -n -o 'id="[0-9]*" aria-roledescription="slide"[^>]*data-label="[^"]*"' docs/deck/asimov-2.0-deck.html
  ```
- **Speaker notes** are in the slide's `data-speaker-notes="..."` attribute.
- **Images** must be embedded as `data:` URIs. The page's security policy blocks images loaded from the web.
- **Adding or removing a slide:** copy a whole `<section>` line, renumber `id` and `aria-label="Slide N of M"` on every slide, and add or remove the matching entry in the `slides` array of the `<script id="deck-motion">` block near the top of `<body>` (it holds each slide's background colour). Then open the file and check it.
- Text still to fill in: `[TEAM]` (slide 1), `[TEAMMATE TO FILL]` (slide 20), and slides 27–30 and 34–35.

## Working together

- Branch, edit, open a PR into `main`, as for code. Pull `main` before you start.
- Because each slide is one line, two people editing **the same slide** will conflict; different slides merge cleanly. Say in the chat which slides you're taking.
- Numbers on a slide should come from `results/` and cite the run in the footer or the speaker notes, as the existing slides do.
