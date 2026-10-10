"""expect: a prediction on an act, checked by the harness against the stepper's own result."""
import pytest

from sage.gateway import expectations as ex

PAYLOAD = {  # the shape being_board_step.py returns (legion-being trace, 2026-10-10)
    "game": "ar25-0c556536",
    "probes": [{"move": 36, "action": "ACTION3", "cells_changed": 670},
               {"move": 37, "action": "ACTION3", "cells_changed": 0}],
    "looks": [{"window": {"x": [18, 30], "y": [15, 27]},
               "rows": ["y=15:  9  9  5  9  9  9  9  9  9  9  9  9 10",
                        "y=16:  9  9  9  9  9  9  9  9  9  9  9  9 10"]}],
    "state_after": {"state": "GameState.NOT_FINISHED", "levels_completed": 1},
}


def outcomes(preds):
    return [c["outcome"] for c in ex.check_game(ex.parse(preds), PAYLOAD)]


def test_each_predicate_kind_is_checked_against_the_result():
    assert outcomes([{"probe": 1, "changes": True}, {"probe": 2, "changes": True},
                     {"levels_completed": 1}, {"levels_completed": 2},
                     {"state": "NOT_FINISHED"}, {"state": "WIN"},
                     {"cell": [20, 15], "value": 5}, {"cell": [20, 15], "value": 9}]) == [
        "confirmed", "contradicted", "confirmed", "contradicted",
        "confirmed", "contradicted", "confirmed", "contradicted"]


def test_what_cannot_be_checked_is_said_not_guessed():
    got = ex.check_game(ex.parse([{"cell": [2, 2], "value": 5}, {"probe": 9, "changes": True},
                                  {"vibe": "good"}]), PAYLOAD)
    assert [c["outcome"] for c in got] == ["unverifiable", "unverifiable", "unrecognized"]
    assert "no LOOK" in got[0]["observed"]


def test_the_summary_names_every_contradiction():
    s = ex.summary(ex.check_game(ex.parse('[{"probe": 2, "changes": true}, {"levels_completed": 1}]'), PAYLOAD))
    assert s.startswith("Your expectations: 1 confirmed, 1 CONTRADICTED") and "0 cells changed" in s


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"text"', "[" + ",".join(['{"state":"WIN"}'] * 17) + "]"])
def test_malformed_expect_is_refused_with_a_reason(raw):
    with pytest.raises(ValueError):
        ex.parse(raw)


def test_the_game_schema_offers_expect_and_the_trial_can_withdraw_it():
    from sage.gateway import toolset
    specs = toolset.specs({}, None)
    game = [s for s in specs if s["function"]["name"] == "game"][0]
    assert "expect" in game["function"]["parameters"]["properties"]
    # a fresh call is unaffected by a pop on an earlier one (the trial removes it per beat)
    game["function"]["parameters"]["properties"].pop("expect")
    again = [s for s in toolset.specs({}, None) if s["function"]["name"] == "game"][0]
    assert "expect" in again["function"]["parameters"]["properties"]


def test_main_records_the_arm_and_strips_expect_off_arm():
    import inspect
    from sage.gateway import heartbeat
    src = inspect.getsource(heartbeat.main)
    assert 'trial_arm(instance, instance / "heartbeats.jsonl", "expect_trial")' in src
    assert 'record["expect_trial"] = _expect_arm' in src and '.pop("expect", None)' in src
