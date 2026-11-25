fintext-ui
==========

The demonstration surface for the scoring service: paste a news item, pick a consumer
profile, see the scores with the triggering spans highlighted.

React is the controller only. It holds the form state, calls POST /v1/signal and renders
what comes back — no scoring logic of its own, no privileged access, so it cannot
display anything the API does not return.

The gate's call leads the results, because that is what a trading system consumes. It is
set in three weights on purpose: buy and sell read loud, a review that needs a person
reads loud in amber, and a routine review is quiet and small. Most of the stream is that
last one, and if it shouted as loudly as the rest the page would be useless.

Develop
-------

    npm install
    npm run dev          # http://localhost:5173, hot reload

The dev server proxies /v1 and /healthz to the Python service on port 8080, so run
make serve from the repository root in another terminal.

Build
-----

    npm run build

Output goes to ../src/fintext/static/, which the Python package ships as package data
and serves at /ui. The built assets are committed, so make serve works on a machine with
no Node installed. Rebuild and commit whenever you change anything under src/.

Design language
---------------

Everything is set by a handful of tokens at the top of styles.css:

- One ink, no second hue. A near-black ink (#070708), and every grey is that ink at
  reduced alpha rather than a separate scale. Light and dark stay consistent by
  construction: change the ink and the page follows.
- The accent is the ink used as a surface. Solid black fills carrying paper-coloured
  type — the status band, the primary button, the signal chips. It inverts with the
  theme, so the same rules give a white accent on the dark page with no second set of
  tokens.
- Colour is reserved for verdicts. Green, red and blue appear only on the words that
  carry a judgement, so the eye lands there first.
- Highlights are distinguished by hue, not weight. A solid fill in running text reads as
  redaction, so marks are a light wash plus a coloured rule underneath: red for toxicity
  evidence, ink for scam patterns, blue for the checkable claim.
- Type does the emphasis. Verdicts are large with tight negative tracking (-0.04em) and
  a 1.0 line height; small labels run the other way — uppercase, 0.13em tracking, muted.
  Numbers and machine values are monospace, prose is not.
- Pills for anything interactive, a generous radius on surfaces, hairline borders in
  ink-alpha rather than a grey.

System fonts throughout: the page has to render with no network, which rules out a
webfont. The stack picks a grotesk where one is installed and the tracking is tuned for
it.

Layout
------

    index.html              the HTML shell
    src/
      main.jsx              mounts the app
      App.jsx               the controller: request state, response state, errors
      api.js                the only place that talks to the Python service
      samples.js            the example items behind the sample buttons
      styles.css            plain CSS, no framework, no CSS-in-JS
      components/           one presentational component per result card
