Lexicons
========

The lexicon floor runs after the model: any match against a high-severity list raises
the toxic flag regardless of model score (PRD 4.2, 6.5).

File format
-----------

Tab-separated, # starts a comment, blank lines ignored:

    term<TAB>subtype<TAB>severity<TAB>notes

- term: matched case-insensitively against the normalised text, on word boundaries for
  scripts that have them. A term containing a space is matched as a phrase.
- subtype: one of hate, harassment, profanity, threat.
- severity: high forces the toxic flag; medium contributes to the subtype score but
  does not override the model.

Curation policy (PRD 6.5)
-------------------------

- English is curated in-house. Entries that are slurs are held in a controlled list
  managed by the operations team and are not committed to this repository; the committed
  file carries __slur_placeholder_N__ tokens so the loading, versioning and matching
  paths stay exercised in CI and in tests. Point FINTEXT_LEXICON_DIR at the controlled
  directory in a real deployment.
- Arabic is not curated here. The developer does not read Arabic. The Arabic file is a
  versioned copy of a lexicon published with the OSACT literature; work on this side is
  limited to versioning and integration, and is verified during native-speaker review
  hours (PRD 6.5, 11).

Versioning
----------

A lexicon change is a release. Bump the .vN suffix in the filename, add a CHANGELOG.txt
entry with the row count and digest, and update configs/default.toml. The version string
appears in every response's lineage.lexicon_versions, so a flag can always be traced to
the list that raised it.
