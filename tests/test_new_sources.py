"""The publisher-sourced datasets, partial subtype labels, and the dialect slice."""

from __future__ import annotations

import json
import tarfile
import zipfile

import pytest

from fintext import catalogue, corpora
from fintext.training import TrainingConfig, compute_metrics


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(corpora, "RAW_ROOT", tmp_path)
    return tmp_path


def write(root, source, name, content):
    path = root / source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.write_bytes if isinstance(content, bytes) else path.write_text)(content)
    return path


class TestOsact:
    def test_osact4_reads_both_published_parts(self, raw):
        write(
            raw,
            "osact4",
            "OSACT2020-sharedTask-train.txt",
            "tweet one<LF>more\tOFF\tHS\ntweet two\tNOT_OFF\tNOT_HS\n",
        )
        write(raw, "osact4", "OSACT2020-sharedTask-dev.txt", "tweet three\tOFF\tNOT_HS\n")
        rows = corpora._rows_osact4(None)
        assert len(rows) == 3
        first = next(r for r in rows if r["text"] == "tweet one more")  # <LF> unfolded
        assert first["labels"] == {"toxicity": 1.0, "toxicity_subtypes": [1.0, None, None, None]}
        assert all(r["language"] == "ar" for r in rows)

    def test_osact5_leaves_harassment_unlabelled(self, raw):
        """OSACT5 has no harassment label."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "tweet": ["a", "b"],
                "off": ["OFF", "NOT_OFF"],
                "hs": ["HS3", "NOT_HS"],
                "vlg": ["VLG", "NOT_VLG"],
                "vio": ["NOT_VIO", "NOT_VIO"],
            }
        )
        for name in ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"):
            (raw / "osact5").mkdir(exist_ok=True)
            pq.write_table(table, raw / "osact5" / name)
        rows = corpora._rows_osact5(None)
        hit = next(r for r in rows if r["text"] == "a")
        assert hit["labels"]["toxicity_subtypes"] == [1.0, None, 1.0, 0.0]
        assert hit["labels"]["toxicity"] == 1.0


class TestDialectSlice:
    def test_lhsab_is_its_own_language_and_evaluation_only(self, raw):
        write(raw, "lhsab", "L-HSAB", "Tweet\tClass\nx\tnormal\ny\tabusive\nz\thate\n")
        rows = corpora._rows_lhsab(None)
        assert [r["labels"]["toxicity"] for r in rows] == [0.0, 1.0, 1.0]
        assert {r["language"] for r in rows} == {"ar-lev"}
        assert "lhsab" in corpora.EVAL_ONLY_LOADERS
        assert catalogue.TRAIN not in catalogue.get("lhsab").uses

    def test_a_dialect_is_reported_at_the_arabic_threshold_and_moves_nothing(self):
        msa = {"toxicity_true_ar": [1, 0, 1, 0], "toxicity_score_ar": [0.9, 0.2, 0.8, 0.1]}
        with_slice = {**msa, "toxicity_true_ar-lev": [1, 0], "toxicity_score_ar-lev": [0.7, 0.6]}
        config = TrainingConfig(recall_floor=0.5, precision_floor=0.5)
        alone, both = compute_metrics(msa, msa, config), compute_metrics(msa, with_slice, config)
        assert both["toxicity_f1_ar"] == alone["toxicity_f1_ar"]
        assert "toxicity_auroc_ar-lev" in both
        assert "ar-lev" not in str(both["thresholds"])


class TestEnglishSources:
    def test_hatexplain_drops_posts_with_no_majority(self, raw):
        posts = {
            "1": {
                "post_tokens": ["a", "b"],
                "annotators": [{"label": vote} for vote in ("hatespeech", "hatespeech", "normal")],
            },
            "2": {
                "post_tokens": ["c"],
                "annotators": [{"label": vote} for vote in ("hatespeech", "offensive", "normal")],
            },
        }
        write(raw, "hatexplain", "dataset.json", json.dumps(posts))
        rows = corpora._rows_hatexplain(None)
        assert len(rows) == 1
        assert rows[0]["labels"] == {"toxicity": 1.0, "toxicity_subtypes": [1.0, None, None, None]}

    def test_wikipedia_talk_takes_the_majority_of_annotators(self, raw):
        write(
            raw,
            "wikipedia_talk",
            "toxicity_annotated_comments.tsv",
            "rev_id\tcomment\tyear\tlogged_in\tns\tsample\tsplit\n"
            "1.0\tyou NEWLINE_TOKEN idiot\t2015\tTrue\tuser\trandom\ttrain\n"
            "2.0\tthanks for the edit\t2015\tTrue\tuser\trandom\ttrain\n",
        )
        write(
            raw,
            "wikipedia_talk",
            "toxicity_annotations.tsv",
            "rev_id\tworker_id\ttoxicity\ttoxicity_score\n"
            "1.0\t1\t1\t-1.0\n1.0\t2\t1\t-2.0\n1.0\t3\t0\t0.0\n"
            "2.0\t1\t0\t1.0\n2.0\t2\t0\t0.0\n",
        )
        rows = {r["text"]: r["labels"]["toxicity"] for r in corpora._rows_wikipedia_talk(None)}
        assert rows == {"you   idiot": 1.0, "thanks for the edit": 0.0}

    def test_isot_hides_the_wire_service(self, raw):
        """Every real ISOT article is Reuters."""
        bundle = raw / "isot_fake_news" / "News-_dataset.zip"
        bundle.parent.mkdir(parents=True)
        with zipfile.ZipFile(bundle, "w") as z:
            z.writestr(
                "True.csv",
                'title,text,subject,date\n"Senate votes","WASHINGTON (Reuters) - '
                'The Senate voted, a spokesman told Reuters.",politics,2017\n',
            )
            z.writestr(
                "Fake.csv",
                'title,text,subject,date\n"You wont believe","Shocking news today.",News,2017\n',
            )
        rows = corpora._rows_isot(None)
        assert not any("Reuters" in r["text"] for r in rows)
        assert {r["labels"]["misleading"] for r in rows} == {0.0, 1.0}

    def test_fakenewsnet_labels_by_file(self, raw):
        for name, title in (
            ("politifact_fake", "fake one"),
            ("politifact_real", "real one"),
            ("gossipcop_fake", "fake two"),
            ("gossipcop_real", "real two"),
        ):
            write(raw, "fakenewsnet", f"{name}.csv", f"id,news_url,title,tweet_ids\n1,u,{title},\n")
        rows = {r["text"]: r["labels"]["misleading"] for r in corpora._rows_fakenewsnet(None)}
        assert rows == {"fake one": 1.0, "real one": 0.0, "fake two": 1.0, "real two": 0.0}

    def test_pheme_reads_source_tweets_from_the_archive(self, raw):
        staging = raw / "staging" / "pheme-rnr-dataset" / "ottawashooting"
        for kind, text in (("rumours", "gunman spotted"), ("non-rumours", "police statement")):
            tweet = staging / kind / "1" / "source-tweet" / "1.json"
            tweet.parent.mkdir(parents=True)
            tweet.write_text(json.dumps({"text": text}))
        archive = raw / "pheme" / "pheme-rnr-dataset.tar.bz2"
        archive.parent.mkdir(parents=True)
        with tarfile.open(archive, "w:bz2") as t:
            t.add(raw / "staging" / "pheme-rnr-dataset", arcname="pheme-rnr-dataset")
        rows = {r["text"]: r["labels"]["misleading"] for r in corpora._rows_pheme(None)}
        assert rows == {"gunman spotted": 1.0, "police statement": 0.0}


class TestCatalogueAgreement:
    def test_every_training_source_is_cleared_and_has_a_mirror(self):
        catalogue.check_training_sources(list(corpora.LOADERS))
        for key in list(corpora.LOADERS) + list(corpora.EVAL_ONLY_LOADERS):
            assert corpora.MIRRORS[key].path

    def test_the_real_osact4_is_the_default(self):
        assert corpora.LOADERS["osact4"] is corpora._rows_osact4
        assert "SUBSTITUTION" in corpora.SUBSTITUTE_MIRRORS["osact4"].note


class TestPartialSubtypes:
    def test_unlabelled_subtypes_do_not_reach_the_loss(self):
        torch = pytest.importorskip("torch")
        from fintext.training import multitask_loss

        logits = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        outputs = {"toxicity_subtypes": logits}
        masks = {"toxicity_subtypes": torch.tensor([True])}
        partial = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # Only hate is labelled (and correct), so the loss is small. Counting the three
        # unlabelled zeros would make it large.
        masks["toxicity_subtypes_elements"] = torch.tensor([[True, False, False, False]])
        labelled_only, _ = multitask_loss(outputs, {"toxicity_subtypes": partial}, masks)
        masks["toxicity_subtypes_elements"] = torch.tensor([[True, True, True, True]])
        all_four, _ = multitask_loss(outputs, {"toxicity_subtypes": partial}, masks)
        assert float(labelled_only) < 0.01 < float(all_four)


class TestSentimentWeight:
    def test_the_weight_scales_only_the_sentiment_term(self):
        torch = pytest.importorskip("torch")
        from fintext.training import TrainingConfig, multitask_loss

        outputs = {"sentiment": torch.tensor([[0.2, 0.1, 0.0]])}
        targets = {"sentiment": torch.tensor([1])}
        masks = {"sentiment": torch.tensor([True])}
        base, _ = multitask_loss(outputs, targets, masks)
        doubled, _ = multitask_loss(outputs, targets, masks, {"sentiment": 2.0})
        assert float(doubled) == pytest.approx(2 * float(base))
        assert TrainingConfig().sentiment_weight == 2.0


class TestArabicToxicityWeight:
    def test_arabic_rows_count_for_more(self):
        torch = pytest.importorskip("torch")
        from fintext.training import TrainingConfig, multitask_loss

        assert TrainingConfig().arabic_toxicity_weight > 1.0
        # Row 0 (English) is badly wrong, row 1 (Arabic) is nearly right.
        outputs = {"toxicity": torch.tensor([[3.0], [3.0]])}
        targets = {"toxicity": torch.tensor([0.0, 1.0])}
        flat = {"toxicity": torch.tensor([True, True])}
        even, _ = multitask_loss(outputs, targets, flat)
        weighted = flat | {"toxicity_row_weight": torch.tensor([1.0, 1.5])}
        arabic_heavier, _ = multitask_loss(outputs, targets, weighted)
        # More weight on the row that's right pulls the mean loss down.
        assert float(arabic_heavier) < float(even)

    def test_equal_weights_change_nothing(self):
        torch = pytest.importorskip("torch")
        from fintext.training import multitask_loss

        outputs = {"toxicity": torch.tensor([[0.3], [-1.0], [2.0]])}
        targets = {"toxicity": torch.tensor([0.0, 1.0, 1.0])}
        flat = {"toxicity": torch.tensor([True, True, True])}
        ones = flat | {"toxicity_row_weight": torch.tensor([1.5, 1.5, 1.5])}
        assert float(multitask_loss(outputs, targets, flat)[0]) == pytest.approx(
            float(multitask_loss(outputs, targets, ones)[0])
        )


class TestTranslatedSentiment:
    """Arabic copies of the English sentiment sets, train only (PRD 7)."""

    @pytest.fixture
    def sources(self, monkeypatch):
        english = [
            {"text": f"english {i}", "language": "en", "labels": {"sentiment": "positive"}}
            for i in range(200)
        ]
        arabic = [
            {
                "text": f"arabic {i}",
                "language": "ar",
                "labels": {"sentiment": "positive"},
                "source_text": f"english {i}",
                "source": "financial_phrasebank",
            }
            for i in range(200)
        ]
        monkeypatch.setitem(corpora.LOADERS, "financial_phrasebank", lambda limit: english)
        monkeypatch.setitem(
            corpora.TRAIN_ONLY_LOADERS, "translated_sentiment_ar", lambda limit: arabic
        )

    def test_a_translation_trains_only_alongside_its_original(self, sources):
        """If the Arabic copy of an English test sentence were in training, the shared encoder
        would have seen that test item, and the English score would rise for the wrong reason.
        """
        split, prov = corpora.load_rows(
            ("financial_phrasebank",), train_only=("translated_sentiment_ar",)
        )
        english_train = {r["text"] for r in split["train"] if r["language"] == "en"}
        arabic_train = [r for r in split["train"] if r["language"] == "ar"]
        assert arabic_train
        assert all(r["source_text"] in english_train for r in arabic_train)
        record = prov["sources"]["translated_sentiment_ar"]
        assert record["rows"] + record["dropped_matches_dev_or_test"] == 200

    def test_translations_never_reach_dev_or_test(self, sources):
        split, _ = corpora.load_rows(
            ("financial_phrasebank",), train_only=("translated_sentiment_ar",)
        )
        for name in ("dev", "test"):
            assert not any(r["language"] == "ar" for r in split[name])

    def test_without_it_the_split_is_unchanged(self, sources):
        plain, _ = corpora.load_rows(("financial_phrasebank",))
        with_it, _ = corpora.load_rows(
            ("financial_phrasebank",), train_only=("translated_sentiment_ar",)
        )
        english = sorted(r["text"] for r in with_it["train"] if r["language"] == "en")
        assert english == sorted(r["text"] for r in plain["train"])
        assert with_it["dev"] == plain["dev"] and with_it["test"] == plain["test"]


class TestBrokenTranslations:
    """Real failures from the translated set, and a good translation for contrast."""

    @pytest.mark.parametrize(
        "arabic,english,reason",
        [
            ("USSS", "$amzn breaking", "more Latin than Arabic"),
            (
                "اتجاه صرف صرف صرف صرف صرف صرف صرف صرف مقابل متوسط 110 وهبوط",
                "$FB trend, hits mid 110's and drops",
                "repeated",
            ),
            ("خروج", "$DARA breaking out of its range after the earnings report", "30%"),
            ("", "Profit rose", "empty"),
        ],
    )
    def test_a_broken_translation_is_named(self, arabic, english, reason):
        assert reason in corpora.translation_problem(arabic, english)

    def test_a_good_translation_passes(self):
        assert (
            corpora.translation_problem(
                "وارتفعت أرباح التشغيل إلى 13.1 مليون يورو من 8.7 مليون يورو",
                "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn",
            )
            is None
        )

    def test_a_ticker_inside_arabic_text_is_fine(self):
        assert (
            corpora.translation_problem(
                "ارتفع سهم AAPL بعد إعلان الأرباح الفصلية القوية", "AAPL rose after strong earnings"
            )
            is None
        )

    def test_the_loader_drops_them(self, raw):
        good = {
            "text": "ارتفعت الأرباح بشكل كبير هذا العام",
            "language": "ar",
            "labels": {"sentiment": "positive"},
            "source_text": "Profits rose sharply this year",
        }
        broken = {**good, "text": "USSS", "source_text": "$amzn breaking"}
        path = raw / "translated_sentiment_ar" / "rows.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (good, broken)))
        assert [r["text"] for r in corpora._rows_translated_sentiment(None)] == [good["text"]]


class TestNoSilentFailures:
    """A Colab run once trained with FiQA missing and PHEME at 8% of its size, and nothing in the
    output said so.
    """

    def test_a_source_that_loads_nothing_stops_the_run(self, monkeypatch):
        monkeypatch.setitem(corpora.LOADERS, "liar", lambda limit: [])
        with pytest.raises(corpora.CorpusError, match="liar loaded 0 rows"):
            corpora.load_rows(("liar",))

    def test_a_refused_fiqa_download_says_what_to_do(self, monkeypatch):
        def refuse(path, split="train"):
            raise RuntimeError("is a gated dataset on the Hub")

        monkeypatch.setattr(corpora, "_load", refuse)
        with pytest.raises(corpora.CorpusError, match="gated") as caught:
            corpora._rows_fiqa(None)
        assert "huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification" in str(caught.value)

    def test_pheme_ignores_a_half_unpacked_folder(self, raw):
        """Leftovers from an interrupted unpack must not stand in for the data."""
        leftover = raw / "pheme" / "pheme-rnr-dataset" / "ottawashooting" / "rumours" / "9"
        (leftover / "source-tweet").mkdir(parents=True)
        (leftover / "source-tweet" / "9.json").write_text(json.dumps({"text": "stale"}))
        staging = raw / "staging" / "pheme-rnr-dataset" / "sydneysiege"
        for kind, text in (("rumours", "one"), ("non-rumours", "two")):
            tweet = staging / kind / "1" / "source-tweet" / "1.json"
            tweet.parent.mkdir(parents=True)
            tweet.write_text(json.dumps({"text": text}))
        with tarfile.open(raw / "pheme" / "pheme-rnr-dataset.tar.bz2", "w:bz2") as t:
            t.add(raw / "staging" / "pheme-rnr-dataset", arcname="pheme-rnr-dataset")
        assert {r["text"] for r in corpora._rows_pheme(None)} == {"one", "two"}


class TestNativeArabicSentiment:
    """ArSarcasm-v2 trains only, and never on a tweet that is held out."""

    @pytest.fixture
    def sources(self, monkeypatch):
        english = [
            {"text": f"english {i}", "language": "en", "labels": {"sentiment": "neutral"}}
            for i in range(100)
        ]
        held = [
            {
                "text": f"تغريدة رقم {chr(0x0628 + i)} عن السوق",
                "language": "ar",
                "labels": {"sentiment": "neutral"},
            }
            for i in range(20)
        ]
        native = [
            {
                "text": f"تغريدة جديدة {chr(0x0628 + i)}",
                "language": "ar",
                "labels": {"sentiment": "positive"},
            }
            for i in range(20)
        ]
        # The same tweet as a held-out one, with a handle, a link and elongation.
        native.append(
            {
                "text": "@someone تغريدددة رقم " + chr(0x0628) + " عن السوق https://t.co/x",
                "language": "ar",
                "labels": {"sentiment": "neutral"},
            }
        )

        # Fresh copies per call, as a real loader returns: load_rows shuffles and annotates the
        # rows it's given.
        def fresh(rows):
            return lambda limit: [dict(r) for r in rows]

        monkeypatch.setitem(corpora.LOADERS, "financial_phrasebank", fresh(english))
        monkeypatch.setitem(corpora.EVAL_ONLY_LOADERS, "arsas", fresh(held))
        monkeypatch.setitem(corpora.TRAIN_ONLY_LOADERS, "arsarcasm_v2", fresh(native))

    def test_a_held_out_tweet_is_never_trained_on(self, sources):
        split, prov = corpora.load_rows(
            ("financial_phrasebank",), eval_sources=("arsas",), train_only=("arsarcasm_v2",)
        )
        trained = {r["text"] for r in split["train"] if r.get("corpus") == "arsarcasm_v2"}
        assert len(trained) == 20
        assert not any("السوق" in t for t in trained)
        assert prov["sources"]["arsarcasm_v2"]["dropped_matches_dev_or_test"] == 1

    def test_adding_it_leaves_dev_and_test_alone(self, sources):
        without, a = corpora.load_rows(("financial_phrasebank",), eval_sources=("arsas",))
        with_it, b = corpora.load_rows(
            ("financial_phrasebank",), eval_sources=("arsas",), train_only=("arsarcasm_v2",)
        )
        assert without["dev"] == with_it["dev"] and without["test"] == with_it["test"]
        assert a["totals"]["heldout_fingerprint"] == b["totals"]["heldout_fingerprint"]
        assert a["totals"]["train_fingerprint"] != b["totals"]["train_fingerprint"]

    def test_catalogue_allows_training_not_evaluation(self):
        entry = catalogue.get("arsarcasm_v2")
        assert catalogue.TRAIN in entry.uses and catalogue.EVAL not in entry.uses
