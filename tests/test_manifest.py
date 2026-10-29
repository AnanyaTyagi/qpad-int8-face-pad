import pandas as pd
import pytest

from qpad.manifest import (derived_counts, expected_protocol_ids, overall_status, pai_class, parse_oulu_name,
                           parse_protocol_text, protocol_file_split, protocol_split, validate_structure, CheckResult)


def test_parse_valid_name(real_cfg):
    oid = parse_oulu_name("Train_files/3_2_17_4.avi", real_cfg)
    assert (oid.phone, oid.session, oid.subject, oid.pai) == (3, 2, 17, 4)
    assert oid.video_id == "3_2_17_4"


@pytest.mark.parametrize("bad", ["7_1_01_1", "1_4_01_1", "1_1_56_1", "1_1_01_6", "1_1_01", "a_1_01_1", "1-1-01-1"])
def test_parse_invalid_names(real_cfg, bad):
    with pytest.raises(ValueError):
        parse_oulu_name(bad, real_cfg)


def test_pai_mapping(real_cfg):
    assert [pai_class(k, real_cfg) for k in (1, 2, 3, 4, 5)] == ["bona_fide", "print", "print", "replay", "replay"]
    with pytest.raises(ValueError):
        pai_class(6, real_cfg)


def test_protocol2_membership(real_cfg):
    p = lambda s: protocol_split(parse_oulu_name(s, real_cfg), real_cfg)  # noqa: E731
    assert p("1_1_01_1") == "train" and p("1_1_01_2") == "train" and p("1_1_01_4") == "train"
    assert p("1_1_01_3") is None and p("1_1_01_5") is None          # Printer 2 / Display 2 not in train
    assert p("1_1_21_2") == "dev" and p("1_1_35_5") is None
    assert p("1_1_36_3") == "test" and p("1_1_55_5") == "test"
    assert p("1_1_36_2") is None and p("1_1_36_4") is None          # Printer 1 / Display 1 not in test


def test_expected_counts_match_table(real_cfg):
    d = derived_counts(real_cfg)
    assert d == {"train": {"bona_fide": 360, "print": 360, "replay": 360},
                 "dev": {"bona_fide": 270, "print": 270, "replay": 270},
                 "test": {"bona_fide": 360, "print": 360, "replay": 360}}
    exp = expected_protocol_ids(real_cfg)
    assert len(exp) == 2970 and exp["video_id"].is_unique
    subj = exp["video_id"].str.split("_").str[2].astype(int)
    assert set(subj[exp["split"] == "train"]) == set(range(1, 21))
    assert set(subj[exp["split"] == "dev"]) == set(range(21, 36))
    assert set(subj[exp["split"] == "test"]) == set(range(36, 56))


def test_protocol_text_parsing():
    assert parse_protocol_text("+1,1_1_01_1\n-1,1_1_01_2\n\n") == ["1_1_01_1", "1_1_01_2"]
    assert protocol_file_split("OULU/Protocols/Protocol_2/Train.txt", "Protocol_2") == "train"
    assert protocol_file_split("OULU/Protocols/Protocol_3/Train_1.txt", "Protocol_2") is None


def _videos(rows):
    base = {"parse_error": "", "in_protocol": True, "annotation_candidates": 1, "member_path": "x"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_leakage_and_duplicates_detected(real_cfg):
    v = _videos([
        {"video_id": "1_1_01_1", "subject": 1, "archive_key": "train", "subject_split": "train", "split": "train", "class": "bona_fide"},
        {"video_id": "1_1_01_1", "subject": 1, "archive_key": "dev", "subject_split": "train", "split": "train", "class": "bona_fide"},
    ])
    checks = {c.name: c for c in validate_structure(v, real_cfg)}
    assert checks["no_duplicate_video_ids"].status == "fail"
    assert checks["no_subject_leakage"].status == "fail"
    assert checks["archive_matches_subject_split"].status == "fail"
    assert checks["no_missing_protocol_videos"].status == "fail"
    assert checks["stated_counts_match_membership_rule"].status == "pass"


def test_overall_status_requires_all_required_checks():
    ok = CheckResult("a", "pass", True, "")
    assert overall_status([ok]) == "PASS"
    assert overall_status([ok, CheckResult("b", "not_checked", True, "")]) == "INCOMPLETE"
    assert overall_status([ok, CheckResult("b", "not_checked", False, "")]) == "PASS"
    assert overall_status([ok, CheckResult("c", "fail", True, "")]) == "FAIL"
