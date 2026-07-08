#!/usr/bin/env python3
"""Salience filtering — the high-bandwidth perceptual stream is mostly redundant;
only a tiny fraction is signal. This extracts it, SNARC-style, so that what reaches
the being's context (its journal, its raising digest) is the salient fraction, not
the raw torrent. For a mind whose identity lives in context, the filter *is* the
sensory curriculum.

SNARC-lite over the perceptual state:
  - Surprise  : prediction error vs a running expectation (EMA) of the stream.
  - Novelty   : 1 - habituation. Repeated event-signatures fade; new ones stand out.
  - Arousal   : raw intensity (motion + rotation rate).
  - Conflict  : reafference disagreement — vision says the world moved AND the IMU
                says I moved, so the cause is ambiguous. (Falls out of embodiment.)
  - Reward    : deferred (no goals yet).

A moment is "salient" when the blended score clears a threshold. Habituation means
88 near-identical "I was handled" events collapse to a couple of salient ones plus
a fade to normal — not 88 signals.
"""
from __future__ import annotations


class SalienceFilter:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.exp_motion = 0.0    # expected max motion (EMA)
        self.exp_self = 0.0      # expected self-motion presence (EMA of 0/1)
        self.exp_trust = 0.85    # expected view quality (EMA)
        self.hab: dict = {}      # event-signature -> habituation level (0..1)

    def score(self, state: dict) -> dict:
        cams = state["cameras"]; prop = state.get("proprioception", {})
        motion = max(cams["0"]["motion"], cams["1"]["motion"])
        self_mot = 1.0 if prop.get("self_motion") in ("moving", "rotating") else 0.0
        trust = min(cams["0"]["trust"], cams["1"]["trust"])
        gyro = prop.get("gyro_mag", 0.0)

        # Surprise: how far the moment departs from what we've come to expect.
        surprise = min(1.0, abs(motion - self.exp_motion)
                            + abs(self_mot - self.exp_self)
                            + max(0.0, self.exp_trust - trust))
        # Conflict: reafference ambiguity — visual change AND self-motion at once.
        conflict = 1.0 if (self_mot > 0.5 and motion > 0.15) else 0.0
        # Arousal: raw intensity.
        arousal = min(1.0, motion + gyro / 120.0)
        # Novelty via habituation on a coarse signature of the moment.
        sig = (motion > 0.15, prop.get("self_motion", "?"), "clear" if trust > 0.6 else "murky")
        h = self.hab.get(sig, 0.0)
        novelty = 1.0 - h

        # Blend → salience. Novelty gates arousal (an intense-but-familiar moment is
        # not salient); surprise and conflict contribute directly.
        salience = min(1.0, 0.45 * surprise + 0.30 * (novelty * arousal) + 0.25 * conflict)

        # Update state: habituate this signature, slowly forget the rest, track expectations.
        for k in list(self.hab):
            self.hab[k] *= 0.985
        self.hab[sig] = min(1.0, h + 0.12)
        self.exp_motion = 0.9 * self.exp_motion + 0.1 * motion
        self.exp_self = 0.9 * self.exp_self + 0.1 * self_mot
        self.exp_trust = 0.9 * self.exp_trust + 0.1 * trust

        return {
            "salience": round(salience, 3),
            "salient": salience >= self.threshold,
            "surprise": round(surprise, 3),
            "novelty": round(novelty, 3),
            "arousal": round(arousal, 3),
            "conflict": conflict,
        }
