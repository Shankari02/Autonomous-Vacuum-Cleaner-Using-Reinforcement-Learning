"""Fuzzy logic controller for vacuum decision support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


@dataclass
class FuzzyController:
    """Simple Mamdani-style fuzzy inference system."""

    def fuzzify_dirt(self, dirt_level: float) -> Dict[str, float]:
        return {
            "low": trapezoidal(dirt_level, -0.1, 0.0, 0.8, 1.5),
            "medium": triangular(dirt_level, 0.8, 1.8, 2.7),
            "high": trapezoidal(dirt_level, 2.0, 2.6, 3.0, 3.5),
        }

    def fuzzify_battery(self, battery_level: float) -> Dict[str, float]:
        return {
            "low": trapezoidal(battery_level, -1, 0, 20, 40),
            "medium": triangular(battery_level, 25, 50, 75),
            "high": trapezoidal(battery_level, 60, 80, 100, 110),
        }

    def fuzzify_distance(self, distance: float) -> Dict[str, float]:
        return {
            "near": trapezoidal(distance, -1, 0, 1, 4),
            "far": trapezoidal(distance, 2, 5, 20, 25),
        }

    def infer(self, dirt_level: float, battery_level: float, charger_distance: float) -> Dict[str, float]:
        dirt = self.fuzzify_dirt(dirt_level)
        battery = self.fuzzify_battery(battery_level)
        distance = self.fuzzify_distance(charger_distance)
        output = {"clean": 0.0, "move": 0.0, "recharge": 0.0}

        def apply(rule_strength: float, action: str) -> None:
            output[action] = max(output[action], rule_strength)

        # Core rules
        apply(min(dirt["high"], battery["high"]), "clean")
        apply(min(dirt["high"], battery["medium"]), "clean")
        apply(min(dirt["medium"], battery["high"]), "clean")
        apply(min(battery["low"], distance["near"]), "recharge")
        apply(min(battery["low"], distance["far"]), "move")
        apply(min(dirt["low"], battery["high"]), "move")
        apply(min(dirt["low"], battery["medium"], distance["near"]), "move")
        apply(min(dirt["medium"], battery["low"], distance["near"]), "recharge")
        apply(min(dirt["high"], battery["low"], distance["far"]), "move")
        apply(min(dirt["low"], battery["low"], distance["near"]), "recharge")

        total = sum(output.values())
        if total <= 0:
            return {"clean": 0.34, "move": 0.33, "recharge": 0.33}
        return {key: value / total for key, value in output.items()}

    def preferred_mode(self, dirt_level: float, battery_level: float, charger_distance: float) -> str:
        scores = self.infer(dirt_level, battery_level, charger_distance)
        return max(scores, key=scores.get)
