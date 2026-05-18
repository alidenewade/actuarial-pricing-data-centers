"""Boolean fault-tree evaluator with AND / OR gates and basic events ("Leaf").

Computes top-event failure probability under independence; supports both
exact evaluation and a Monte Carlo cross-check for sanity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Node:
    name: str

    def probability(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Leaf(Node):
    """Basic event with a fixed failure probability."""

    p: float = 0.0

    def probability(self) -> float:
        return float(self.p)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.random(n) < self.p


@dataclass
class AndGate(Node):
    children: Sequence[Node] = field(default_factory=list)

    def probability(self) -> float:
        return float(np.prod([c.probability() for c in self.children]))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.logical_and.reduce([c.sample(n, rng) for c in self.children])


@dataclass
class OrGate(Node):
    children: Sequence[Node] = field(default_factory=list)

    def probability(self) -> float:
        # P(any of independent events) = 1 - prod(1 - p_i)
        return float(1.0 - np.prod([1.0 - c.probability() for c in self.children]))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.logical_or.reduce([c.sample(n, rng) for c in self.children])


@dataclass
class FaultTree:
    """Container around the root node."""

    root: Node

    def top_event_probability(self) -> float:
        return self.root.probability()

    def monte_carlo(self, n: int = 100_000, seed: int | None = None) -> float:
        rng = np.random.default_rng(seed)
        return float(self.root.sample(n, rng).mean())
