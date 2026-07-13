"""Tests for company enrichment (ingestion/enrich_companies.py).

Covers the pure name-matching logic; the SimFin download and DB writes are
exercised by running the module against the live stack.
"""

from __future__ import annotations

from ingestion.enrich_companies import match_document_companies, normalize_company_name


def test_normalization_strips_legal_suffixes_and_state_tags():
    assert normalize_company_name("DELTA AIR LINES INC /DE/") == ("DELTA", "AIR", "LINES")
    assert normalize_company_name("Delta Air Lines") == ("DELTA", "AIR", "LINES")
    assert normalize_company_name("Dave & Buster’s") == ("DAVE", "BUSTERS")
    assert normalize_company_name("Ford Motor Co") == ("FORD", "MOTOR")


def test_forward_prefix_tier_beats_reverse_prefix_ambiguity():
    # 'Dave Inc' matches only via the risky reverse direction; the forward
    # match on Dave & Buster's Entertainment must win alone.
    companies = [
        {"id": 1, "name": "Dave & Buster's Entertainment, Inc."},
        {"id": 2, "name": "Dave Inc."},
    ]
    matched, unmatched = match_document_companies(["Dave & Buster’s"], companies)
    assert matched == {"Dave & Buster’s": 1}
    assert unmatched == []


def test_truncated_doc_name_matches_longer_legal_name():
    companies = [{"id": 7, "name": "JETBLUE AIRWAYS CORP"}]
    matched, _ = match_document_companies(["JetBlue"], companies)
    assert matched == {"JetBlue": 7}


def test_ambiguous_and_unknown_names_stay_unmatched():
    companies = [
        {"id": 1, "name": "Acme Robotics International"},
        {"id": 2, "name": "Acme Robotics Holdings"},
    ]
    matched, unmatched = match_document_companies(
        ["Acme Robotics", "Vistance Networks Inc."], companies
    )
    assert matched == {}
    assert set(unmatched) == {"Acme Robotics", "Vistance Networks Inc."}
