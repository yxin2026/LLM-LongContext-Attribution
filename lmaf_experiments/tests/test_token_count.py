from lmaf.utils.token_count import TokenCounter, make_filler


def test_make_filler_stays_within_budget() -> None:
    counter = TokenCounter()
    filler = make_filler(80, seed=7, counter=counter)
    assert filler
    assert counter.count(filler) <= 80
    assert counter.count(filler) > 40

