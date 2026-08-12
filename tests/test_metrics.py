from telco_multiagent.eval import metrics as M


def test_lcs_and_msc_eap_sas_fixture():
    gold = ["a", "b", "c"]
    agent = ["a", "x", "b", "c"]
    assert M.lcs_len(agent, gold) == 3
    assert M.msc(agent, gold) == 1.0
    assert M.eap(agent, gold) == 0.75
    assert M.sas(agent, gold) == 0.75


def test_msc_missing_step():
    assert M.msc(["a", "c"], ["a", "b", "c"]) == 2 / 3


def test_eap_no_extra():
    assert M.eap(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_eap_empty_agent():
    assert M.eap([], ["a", "b"]) == 1.0


def test_gpc0_gpc1():
    seqs = [["a", "b", "c"], ["a", "b"], ["a", "b", "c", "d"]]
    golds = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
    assert M.gpc0(seqs, golds) == 1 / 3
    # exact + one within Levenshtein 1 (missing c, and extra d)
    assert M.gpc1(seqs, golds) == 1.0


def test_sequence_diversity_and_brs():
    seqs = [["a", "b"], ["a", "b"]]
    assert M.sequence_diversity(seqs) == 0.0
    b = M.brs(1.0, 1.0, 0.0)
    assert abs(b - 1.0) < 1e-9


def test_norm_levenshtein():
    assert M.norm_levenshtein(["a", "b"], ["a", "b"]) == 0.0
    assert M.norm_levenshtein(["a", "b"], ["a", "c"]) == 0.5
