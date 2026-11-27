Data
====

Four kinds of thing live here, and only one of them is committed.

  Folder                 What                                                         In git
  ---------------------- ------------------------------------------------------------ --------
  samples/               tiny fixtures so the README commands run with no downloads   yes
  sources/               every declared source, normalised, one folder each           no
  raw/                   downloaded originals, when --raw-cache is used               no
  interim/, processed/   scratch space for your own work                              no

Why sources/ is not committed
-----------------------------

Financial PhraseBank is CC BY-NC-SA and Jigsaw carries Kaggle's data-use terms. Both
permit you holding the data; neither permits you handing it on. The licence log in
catalogue.py marks them redistributable=False, and a copy sitting in a repository you
send to someone is exactly the thing that flag exists to prevent.

So the folder is git-ignored, and the archive script excludes it. Anyone who needs the
data runs fintext data and gets their own copy from the original publishers, which is
also how the licences expect them to get it.

Rebuilding it
-------------

    fintext data                  # every declared source, default caps applied
    fintext data --limit 2000     # a smaller pass for a quick look
    fintext data --full           # no caps; Jigsaw alone is 159k rows
    fintext data --raw-cache      # put the downloads in data/raw too

Each folder gets rows.jsonl and a manifest.json. The rows are post-mapping: FiQA's
continuous score has already become one of three labels, ArSAS's fourth "Mixed" class
has already been dropped. That is deliberate: the mapping is where the judgement calls
are, so it is the thing worth being able to inspect.

INDEX.json at the top lists every source with its row count and licence.

What is not here
----------------

Trained weights (artifacts/models/) and the decision log (reports/) are elsewhere and
also ignored. The provenance index (the primary sources the corroboration check
searches) is in artifacts/provenance/, and currently holds four records, which is a
fixture rather than a corpus.
