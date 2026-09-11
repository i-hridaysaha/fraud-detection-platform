"""Structural features over the graph that shared attributes draw between transactions.

There is no merchant node in this dataset and no customer identifier either. What there is, is a
multipartite graph: `card1`, `addr1`, `P_emaildomain`, `R_emaildomain`, the normalised
`DeviceInfo` and `dist1` are each shared by many transactions, so every transaction is a node
joined to the value nodes it carries, and two transactions are connected when they share a value
or a chain of them. This module walks that graph forward in time and reads, for each transaction,
what the graph looked like before the transaction's own edges went in.

**The rule is the stage 4 rule, and the graph is where it is hardest to keep.** A connected
component computed over the whole file joins every transaction to every later transaction it will
ever share a value with, which is the future written into a feature. So there is no
`connected_components` call anywhere in the feature path. The graph is a union-find that grows
one timestamp block at a time, and the features for a block are read from the state as it stood
before the block was added. `tests/test_causality.py` rebuilds the graph from scratch using only
rows with `TransactionDT < t` and asserts the two agree. ADR 0024.

**Same-timestamp ties are excluded, as everywhere.** Two transactions at the same instant do not
see each other, whichever nodes they share, because a block is read before it is added. Stage 0
measured 573,349 distinct timestamps across 590,540 rows, so the choice moves real rows.

**The entity's component is anchored on its card1 node.** The ADR 0001 entity key is
`card1 + addr1 + card_start_day`, and `card1` is the null-free component, so every one of an
entity's transactions attaches to the same `card1` node and they are always one component. The
component "the entity currently belongs to" is therefore the component of that node, which is
also defined before the entity's first transaction: the card exists in the graph before the
entity does. When the card has never been seen the component is empty and the count is zero.

**The component fraud rate is label-derived and is treated as such.** Labels come only from a
`LabelSource` built by `label_source`, which is decorated with `train_only` so the frame it reads
has been through `assert_train_only`. A label is counted into its component only once it has
matured: a transaction on day `j` contributes its label to reads on day `d` only when
`j <= d - lag_days`, which is the stage 3 rule from ADR 0014 applied to a component instead of a
level. The lag is an argument rather than a constant here, because its value lives in
`reports/encoding_spec.json` and nothing in this module restates it. A second rate excludes the
entity's own labelled transactions, because ADR 0015 measured what an entity's own labels are
worth and refused them. ADR 0025 records what was measured and what shipped.

**What is null and what is zero** follows stage 4: a count of history is zero when there is no
history, and a comparison or a rate is null when there is nothing to compare or divide by. A
degree is null when the row's own value is null, because there is no node to read.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraud_platform import config, encoders, transforms
from fraud_platform.data_loader import train_only
from fraud_platform.features import _sorted_level_codes, entity_codes

# --- choices ------------------------------------------------------------------------------

DEVICE_COLUMN = encoders.normalised_name("DeviceInfo")

# The six value columns the brief names, in the order the artifact reports them. `DeviceInfo` is
# the normalised column stage 3 built: a raw device string carries a build number, so the raw
# column would split one device model into 1,546 nodes where the normalised one has 281.
LINK_COLUMNS: tuple[str, ...] = (
    "card1",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    DEVICE_COLUMN,
    "dist1",
)

# The node the entity's component is read from. Null-free by measurement (stage 0) and a
# component of the entity key by decision (ADR 0001), so every transaction of an entity attaches
# to it.
ANCHOR_COLUMN = "card1"

# The nodes whose degree is a feature.
DEGREE_COLUMNS: tuple[str, ...] = ("card1", "addr1", DEVICE_COLUMN)

COMPONENT_SIZE = "graph_component_size"
COMPONENT_FRAUD_RATE = "graph_component_fraud_rate"
COMPONENT_FRAUD_RATE_OTHERS = "graph_component_fraud_rate_others"
COMPONENT_N_LABELLED = "graph_component_n_labelled"
ENTITIES_ON_DEVICE = "graph_n_entities_on_device"


def degree_name(column: str) -> str:
    """`graph_degree_card1`. The device node takes the raw column's name, not the `_norm` one."""
    label = "device" if column == DEVICE_COLUMN else column
    return f"graph_degree_{label}"


LABEL_FEATURES: tuple[str, ...] = (
    COMPONENT_FRAUD_RATE,
    COMPONENT_FRAUD_RATE_OTHERS,
    COMPONENT_N_LABELLED,
)

ALL_FEATURES: tuple[str, ...] = (
    COMPONENT_SIZE,
    *(degree_name(column) for column in DEGREE_COLUMNS),
    ENTITIES_ON_DEVICE,
    *LABEL_FEATURES,
)

# The candidate set: what `enabled=True` builds unless a caller asks for more. The component
# features are not in it, and the reason is measured rather than argued: `reports/graph_summary.json`
# records that the six-column graph is one component holding 0.9999 of train transactions by the
# end of the training window, so the size of the component a transaction sits in is a count of
# the transactions before it, and the fraud rate inside it is the lagged training rate. Both are a
# clock. ADR 0024 and ADR 0025. The label-derived features are reachable only by naming them.
CANDIDATE_FEATURES: tuple[str, ...] = (
    *(degree_name(column) for column in DEGREE_COLUMNS),
    ENTITIES_ON_DEVICE,
)

# Whether the graph block is built at all. Off: no graph feature passes the stage 4 ship rule
# (same sign on all three splits with disjoint intervals on each) on its marginal, and the
# artifact records every comparison. The switch is what stage 6 flips to measure the block in a
# model, which is the instrument the marginal comparisons are not.
DEFAULT_ENABLED = False

# The one graph quantity the brief asks for that this module does not emit: distinct addr1 values
# seen for the card so far. That is the typed degree of the card1 node towards addr1 nodes, and
# stage 4 already ships it as `geo_n_distinct_addr1_on_card`. One number, one source; the graph
# reading of it is asserted equal to stage 4's column in tests/test_causality.py.
PROVIDED_BY_STAGE_4: dict[str, str] = {
    "distinct addr1 values seen for this card1 so far": "geo_n_distinct_addr1_on_card",
}

_NULL_LEVEL = -1


# --- configuration -------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConfig:
    """What the graph is built on and which of its features come out.

    `enabled` is the switch stage 6 moves. `features` is the subset of `ALL_FEATURES` that comes
    back when it is on; everything is computed in one pass because the state is shared, and the
    subset is what gets returned. `link_columns` is the graph itself and changing it changes
    every number.
    """

    enabled: bool = DEFAULT_ENABLED
    link_columns: tuple[str, ...] = LINK_COLUMNS
    features: tuple[str, ...] = CANDIDATE_FEATURES

    def with_features(self, *names: str) -> GraphConfig:
        return replace(self, features=tuple(names))

    def switched(self, on: bool) -> GraphConfig:
        return replace(self, enabled=on)

    @property
    def needs_labels(self) -> bool:
        return bool(set(self.features) & set(LABEL_FEATURES))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "link_columns": list(self.link_columns),
            "anchor_column": ANCHOR_COLUMN,
            "features": list(self.features),
            "candidate_features": list(CANDIDATE_FEATURES),
            "all_features": list(ALL_FEATURES),
            "label_features": list(LABEL_FEATURES),
        }


DEFAULT_CONFIG = GraphConfig()


# --- the label source, the only place a label enters ----------------------------------------


@dataclass(frozen=True)
class LabelSource:
    """Train-row labels with the lag they mature under. Built by `label_source` and nowhere else."""

    ids: npt.NDArray[np.int64]
    ts: npt.NDArray[np.int64]
    y: npt.NDArray[np.int64]
    lag_days: int

    @property
    def n_rows(self) -> int:
        return int(self.ids.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_fraud": int(self.y.sum()),
            "lag_days": self.lag_days,
            "max_ts": int(self.ts.max()) if self.ts.size else None,
            "rule": (
                "a transaction on day j contributes its label to a read on day d only when "
                "j <= d - lag_days, and only if it is a train row"
            ),
        }


@train_only
def label_source(frame: pd.DataFrame, lag_days: int) -> LabelSource:
    """The labels the component fraud rate may read, from the train split only.

    Decorated with `train_only`, so the frame goes through `assert_train_only` before a label
    is copied out of it. `lag_days` is required and has no default: the value is stage 3's
    measured choice in `reports/encoding_spec.json`, and a default here would be a second copy
    of it that could drift.
    """
    if lag_days < 0:
        raise ValueError(f"lag_days must be non-negative, got {lag_days}")
    for column in (config.ID_COLUMN, config.TARGET):
        if column not in frame.columns:
            raise KeyError(f"label_source needs {column}")
    order = np.argsort(frame[config.TIME_COLUMN].to_numpy(dtype="int64"), kind="stable")
    return LabelSource(
        ids=frame[config.ID_COLUMN].to_numpy(dtype="int64")[order],
        ts=frame[config.TIME_COLUMN].to_numpy(dtype="int64")[order],
        y=frame[config.TARGET].to_numpy(dtype="int64")[order],
        lag_days=int(lag_days),
    )


# --- the union-find ----------------------------------------------------------------------------


class UnionFind:
    """Disjoint sets over integer nodes, with the per-component counters the features read.

    Four counters travel with a root: nodes in the component, transactions in it, labelled
    transactions in it and the sum of their labels. A node that has not been touched is not
    present and has size zero, so a component is only the nodes that some edge has reached.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.present = [False] * n
        self.n_nodes = [0] * n
        self.n_tx = [0] * n
        self.n_labelled = [0] * n
        self.y_labelled = [0] * n

    def touch(self, node: int, is_transaction: bool) -> None:
        if not self.present[node]:
            self.present[node] = True
            self.n_nodes[node] = 1
            self.n_tx[node] = 1 if is_transaction else 0

    def find(self, node: int) -> int:
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.n_nodes[ra] < self.n_nodes[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.n_nodes[ra] += self.n_nodes[rb]
        self.n_tx[ra] += self.n_tx[rb]
        self.n_labelled[ra] += self.n_labelled[rb]
        self.y_labelled[ra] += self.y_labelled[rb]

    def label(self, node: int, y: int) -> None:
        root = self.find(node)
        self.n_labelled[root] += 1
        self.y_labelled[root] += y


# --- the stream ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphStream:
    """The frame's rows in time order, with every column the graph reads as dense codes."""

    order: npt.NDArray[np.int64]
    ts: npt.NDArray[np.int64]
    ids: npt.NDArray[np.int64]
    entity: npt.NDArray[np.int64]
    codes: dict[str, npt.NDArray[np.int64]]
    n_levels: dict[str, int]

    @property
    def n_rows(self) -> int:
        return int(self.ts.size)


def _coded(series: pd.Series, excluded: Sequence[Any] | None) -> npt.NDArray[np.int64]:
    """Value-ordered dense codes with -1 for a null, and -1 for every excluded value too."""
    if not excluded:
        return _sorted_level_codes(series)
    codes, uniques = pd.factorize(series, sort=True, use_na_sentinel=True)
    coded = np.asarray(codes, dtype="int64")
    hub = pd.Index(uniques).isin(list(excluded))
    coded[(coded >= 0) & hub[np.clip(coded, 0, None)]] = _NULL_LEVEL
    return coded


def make_graph_stream(
    frame: pd.DataFrame,
    cfg: GraphConfig = DEFAULT_CONFIG,
    hubs: Mapping[str, Any] | None = None,
    require_anchor: bool = True,
) -> GraphStream:
    """Sort by time and code every link column.

    A missing link column raises: the graph is the declared one or it is not built. The feature
    build also requires the anchor column among the links and null-free in the data, because
    the entity's component is read from it; a static summary of a graph without card1 is
    allowed to say so. `hubs`, from `fit_hub_values`, names values that link nothing: they are
    coded as absent after the null check, so a hub card reads as a card with no node, which is
    what excluding it means.
    """
    missing = [column for column in cfg.link_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"the graph needs link columns {missing}")
    if require_anchor and ANCHOR_COLUMN not in cfg.link_columns:
        raise ValueError(f"the anchor column {ANCHOR_COLUMN!r} must be one of the link columns")
    for column in (config.ID_COLUMN, config.TIME_COLUMN):
        if column not in frame.columns:
            raise KeyError(f"the graph needs {column}")
    if require_anchor and bool(frame[ANCHOR_COLUMN].isna().any()):
        raise ValueError(
            f"the anchor column {ANCHOR_COLUMN!r} carries a null, and the entity's component "
            "is read from it. Stage 0 measured it null-free; this frame is not"
        )
    excluded = dict(hubs["values"]) if hubs is not None else {}
    ts_all = frame[config.TIME_COLUMN].to_numpy(dtype="int64")
    order = np.argsort(ts_all, kind="stable")
    codes: dict[str, npt.NDArray[np.int64]] = {}
    n_levels: dict[str, int] = {}
    for column in cfg.link_columns:
        coded = _coded(frame[column], excluded.get(column))
        codes[column] = coded[order]
        n_levels[column] = int(coded.max()) + 1 if coded.size else 0
    return GraphStream(
        order=order,
        ts=ts_all[order],
        ids=frame[config.ID_COLUMN].to_numpy(dtype="int64")[order],
        entity=entity_codes(frame)[order],
        codes=codes,
        n_levels=n_levels,
    )


def _time_blocks(ts: npt.NDArray[np.int64]) -> Iterator[tuple[int, int]]:
    """Walk a time-sorted stream in blocks of equal timestamp.

    Every block is read against the state before it and then added whole, so a transaction
    never sees a simultaneous one, whichever nodes they share.
    """
    n = ts.size
    if n == 0:
        return
    boundaries = np.flatnonzero(np.diff(ts)) + 1
    edges = [0, *boundaries.tolist(), n]
    yield from itertools.pairwise(edges)


class IncrementalGraph:
    """The graph as it grows, one timestamp block at a time.

    Node ids are laid out as one contiguous range: the value nodes of each link column in
    config order, then one node per transaction in stream order. `add_block` puts a block's
    edges in; the read methods look at the state without changing it. `build_graph_features`
    reads then adds, in that order, for every block.
    """

    def __init__(self, stream: GraphStream, cfg: GraphConfig, labels: LabelSource | None) -> None:
        self.stream = stream
        self.cfg = cfg
        self.base: dict[str, int] = {}
        total = 0
        for column in cfg.link_columns:
            self.base[column] = total
            total += stream.n_levels[column]
        self.tx_base = total
        self.uf = UnionFind(total + stream.n_rows)
        # Degree of a value node: transactions attached to it. A transaction carries one value
        # per column so an edge is never repeated, and the count is the simple-graph degree.
        self.degree = [0] * total
        self.entities_on_node: dict[int, set[int]] = {}
        self.own_n: dict[int, int] = {}
        self.own_y: dict[int, int] = {}
        self.labels = labels
        self._label_position = 0
        self._label_node: npt.NDArray[np.int64] | None = None
        self._label_day: npt.NDArray[np.int64] | None = None
        self.n_labels_attached = 0
        self.hub_report: dict[str, Any] = {}
        if labels is not None:
            positions = pd.Index(stream.ids).get_indexer(pd.Index(labels.ids))
            found = positions >= 0
            self._label_node = np.where(found, self.tx_base + positions, -1).astype("int64")
            self._label_day = np.floor_divide(labels.ts, config.SECONDS_PER_DAY).astype("int64")

    def value_node(self, column: str, position: int) -> int:
        """Node id of the row's value in `column`, or -1 when the value is null."""
        code = int(self.stream.codes[column][position])
        return -1 if code < 0 else self.base[column] + code

    def mature_labels(self, moment: int) -> None:
        """Count every label that a read at `moment` may see and has not been counted yet.

        A label matures when its day is at least `lag_days` before the read's day and its
        transaction is strictly before the read, so it joins the counters of whatever component
        its transaction sits in now. Later unions carry the counters along.
        """
        if self.labels is None or self._label_node is None or self._label_day is None:
            return
        day = moment // config.SECONDS_PER_DAY
        limit = day - self.labels.lag_days
        ts, y = self.labels.ts, self.labels.y
        position = self._label_position
        n = self.labels.n_rows
        while position < n and ts[position] < moment and self._label_day[position] <= limit:
            node = int(self._label_node[position])
            if node >= 0:
                self.uf.label(node, int(y[position]))
                entity = int(self.stream.entity[node - self.tx_base])
                self.own_n[entity] = self.own_n.get(entity, 0) + 1
                self.own_y[entity] = self.own_y.get(entity, 0) + int(y[position])
                self.n_labels_attached += 1
            position += 1
        self._label_position = position

    def add_block(self, start: int, stop: int) -> None:
        uf = self.uf
        for position in range(start, stop):
            tx = self.tx_base + position
            uf.touch(tx, is_transaction=True)
            for column in self.cfg.link_columns:
                node = self.value_node(column, position)
                if node < 0:
                    continue
                uf.touch(node, is_transaction=False)
                self.degree[node] += 1
                uf.union(tx, node)
            device = self.value_node(DEVICE_COLUMN, position) if DEVICE_COLUMN in self.base else -1
            if device >= 0:
                self.entities_on_node.setdefault(device, set()).add(
                    int(self.stream.entity[position])
                )

    def anchor_root(self, position: int) -> int:
        """Root of the row's card1 component, or -1 when the card has never been seen."""
        node = self.value_node(ANCHOR_COLUMN, position)
        if node < 0 or not self.uf.present[node]:
            return -1
        return self.uf.find(node)

    def component_sizes(self) -> npt.NDArray[np.int64]:
        """Transactions per component, over the components any transaction sits in."""
        counts: dict[int, int] = {}
        for position in range(self.stream.n_rows):
            root = self.uf.find(self.tx_base + position)
            counts[root] = self.uf.n_tx[root]
        return np.array(sorted(counts.values(), reverse=True), dtype="int64")

    def component_node_counts(self) -> npt.NDArray[np.int64]:
        """Nodes per component, value nodes included, in the same order as `component_sizes`."""
        counts: dict[int, tuple[int, int]] = {}
        for position in range(self.stream.n_rows):
            root = self.uf.find(self.tx_base + position)
            counts[root] = (self.uf.n_tx[root], self.uf.n_nodes[root])
        ordered = sorted(counts.values(), key=lambda pair: pair[0], reverse=True)
        return np.array([nodes for _, nodes in ordered], dtype="int64")

    def value_degrees(self, column: str) -> npt.NDArray[np.int64]:
        """Degree of every present value node of one column."""
        base = self.base[column]
        stop = base + self.stream.n_levels[column]
        return np.array(
            [self.degree[node] for node in range(base, stop) if self.uf.present[node]],
            dtype="int64",
        )


# --- the build ------------------------------------------------------------------------------


def _scatter_float(order: npt.NDArray[np.int64], values: list[float]) -> npt.NDArray[np.float64]:
    out = np.empty(len(values), dtype="float64")
    out[order] = np.asarray(values, dtype="float64")
    return out


def _scatter_int(order: npt.NDArray[np.int64], values: list[int]) -> npt.NDArray[np.int32]:
    out = np.empty(len(values), dtype="int32")
    out[order] = np.asarray(values, dtype="int64").astype("int32")
    return out


def build_graph_features(
    frame: pd.DataFrame,
    labels: LabelSource | None = None,
    cfg: GraphConfig = DEFAULT_CONFIG,
    hubs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Every graph feature the config asks for, indexed like `frame`.

    One forward pass over the time-sorted stream. For each timestamp block: mature the labels a
    read at this instant may see, read every feature for every row of the block, then add the
    block's edges. Nothing here is fitted, and the whole stream is the correct input for the
    reason stage 4 gave: a validation row reads earlier training rows as it would at
    authorisation, and a training row cannot read a validation row because validation rows are
    later. The one thing that learns from labels, `LabelSource`, arrives as an argument so it
    has to have been built through the guard.

    When `cfg.enabled` is false the frame comes back with no columns, which is the switch stage 6
    uses for the without-graph run. `hubs`, from `fit_hub_values`, is for the artifact's
    measurements and is not part of the default path: with it, a hub value has no node, so a row
    carrying one reads a degree of zero there and, for the anchor, an empty component.
    """
    unknown = [name for name in cfg.features if name not in ALL_FEATURES]
    if unknown:
        raise ValueError(f"unknown graph features {unknown}, expected from {list(ALL_FEATURES)}")
    if not cfg.enabled:
        return pd.DataFrame(index=frame.index)
    if cfg.needs_labels and labels is None:
        raise ValueError(
            "the component fraud rate needs a LabelSource from label_source; pass one rather "
            "than letting this frame read its own labels"
        )
    for column in DEGREE_COLUMNS:
        if column not in cfg.link_columns:
            raise ValueError(f"degree column {column!r} must be one of the link columns")

    stream = make_graph_stream(frame, cfg, hubs)
    graph = IncrementalGraph(stream, cfg, labels if cfg.needs_labels else None)
    n = stream.n_rows

    component_size: list[int] = [0] * n
    degrees: dict[str, list[float]] = {column: [np.nan] * n for column in DEGREE_COLUMNS}
    entities_on_device: list[float] = [np.nan] * n
    rate: list[float] = [np.nan] * n
    rate_others: list[float] = [np.nan] * n
    n_labelled: list[int] = [0] * n

    for start, stop in _time_blocks(stream.ts):
        graph.mature_labels(int(stream.ts[start]))
        for position in range(start, stop):
            root = graph.anchor_root(position)
            if root >= 0:
                component_size[position] = graph.uf.n_tx[root]
                labelled = graph.uf.n_labelled[root]
                n_labelled[position] = labelled
                if labelled > 0:
                    fraud = graph.uf.y_labelled[root]
                    rate[position] = fraud / labelled
                    entity = int(stream.entity[position])
                    others_n = labelled - graph.own_n.get(entity, 0)
                    if others_n > 0:
                        rate_others[position] = (fraud - graph.own_y.get(entity, 0)) / others_n
            for column in DEGREE_COLUMNS:
                node = graph.value_node(column, position)
                if node >= 0:
                    degrees[column][position] = float(graph.degree[node])
                elif column == ANCHOR_COLUMN:
                    # The card is null-free, so no node here means an excluded hub: a value
                    # that links nothing has degree zero rather than no degree.
                    degrees[column][position] = 0.0
            device = graph.value_node(DEVICE_COLUMN, position)
            if device >= 0:
                entities_on_device[position] = float(len(graph.entities_on_node.get(device, ())))
        graph.add_block(start, stop)

    order = stream.order
    columns: dict[str, npt.NDArray[Any]] = {
        COMPONENT_SIZE: _scatter_int(order, component_size),
        ENTITIES_ON_DEVICE: _scatter_float(order, entities_on_device),
        COMPONENT_FRAUD_RATE: _scatter_float(order, rate),
        COMPONENT_FRAUD_RATE_OTHERS: _scatter_float(order, rate_others),
        COMPONENT_N_LABELLED: _scatter_int(order, n_labelled),
    }
    for column in DEGREE_COLUMNS:
        if column == ANCHOR_COLUMN:
            # The anchor is null-free, checked when the stream was made, so its degree is a count.
            columns[degree_name(column)] = _scatter_int(
                order, [int(value) for value in degrees[column]]
            )
        else:
            columns[degree_name(column)] = _scatter_float(order, degrees[column])

    ordered = [name for name in ALL_FEATURES if name in cfg.features]
    out = pd.DataFrame({name: columns[name] for name in ordered}, index=frame.index)
    # How many labels the stream attached by its end, for the artifact. Carried on the frame's
    # attrs rather than in a column because it is one number about the build, not one per row.
    out.attrs["n_labels_attached"] = graph.n_labels_attached
    return out


def add_graph_features(
    frame: pd.DataFrame,
    labels: LabelSource | None = None,
    cfg: GraphConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """`build_graph_features` concatenated onto the frame."""
    return pd.concat([frame, build_graph_features(frame, labels, cfg)], axis=1)


# --- the catalogue ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphFeatureSpec:
    name: str
    definition: str
    null_meaning: str | None
    label_derived: bool
    state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.name,
            "definition": self.definition,
            "null_meaning": self.null_meaning,
            "label_derived": self.label_derived,
            "requires_running_state": True,
            "state": self.state,
        }


_COMPONENT_STATE = (
    "the union-find over every value node and every transaction node seen so far, with per-root "
    "counters. Unbounded: it grows with the stream and it cannot be windowed, because a "
    "component is a fact about every edge ever added"
)
_DEGREE_STATE = "one counter per value node of the column"


def catalogue(cfg: GraphConfig = DEFAULT_CONFIG) -> tuple[GraphFeatureSpec, ...]:
    """Every feature the module computes, declared, whether or not the config returns it."""
    specs: list[GraphFeatureSpec] = [
        GraphFeatureSpec(
            name=COMPONENT_SIZE,
            definition=(
                "transactions in the connected component of the row's card1 node, over the graph "
                f"of strictly earlier rows linked on {list(cfg.link_columns)}"
            ),
            null_meaning=None,
            label_derived=False,
            state=_COMPONENT_STATE,
        )
    ]
    for column in DEGREE_COLUMNS:
        specs.append(
            GraphFeatureSpec(
                name=degree_name(column),
                definition=f"strictly earlier transactions carrying this row's {column} value",
                null_meaning=(
                    None
                    if column == ANCHOR_COLUMN
                    else f"{column} is null, so there is no node to read"
                ),
                label_derived=False,
                state=_DEGREE_STATE,
            )
        )
    specs.extend(
        [
            GraphFeatureSpec(
                name=ENTITIES_ON_DEVICE,
                definition=(
                    f"distinct entities among the strictly earlier transactions carrying this "
                    f"row's {DEVICE_COLUMN} value"
                ),
                null_meaning=f"{DEVICE_COLUMN} is null",
                label_derived=False,
                state="one set of entity ids per device node",
            ),
            GraphFeatureSpec(
                name=COMPONENT_FRAUD_RATE,
                definition=(
                    "fraud rate over the labelled transactions in the row's card1 component. A "
                    "transaction is labelled when it is a train row and its day is at least "
                    "lag_days before the row's day; the lag is read from "
                    "reports/encoding_spec.json"
                ),
                null_meaning="the component holds no matured train label, or card1 is unseen",
                label_derived=True,
                state=_COMPONENT_STATE,
            ),
            GraphFeatureSpec(
                name=COMPONENT_FRAUD_RATE_OTHERS,
                definition=(
                    "the same rate with the row's own entity's labelled transactions removed "
                    "from both numerator and denominator"
                ),
                null_meaning="no labelled transaction of another entity in the component",
                label_derived=True,
                state=_COMPONENT_STATE + ", plus two counters per entity",
            ),
            GraphFeatureSpec(
                name=COMPONENT_N_LABELLED,
                definition="the count the component fraud rate is taken over",
                null_meaning=None,
                label_derived=True,
                state=_COMPONENT_STATE,
            ),
        ]
    )
    return tuple(specs)


def graph_feature_names(cfg: GraphConfig = DEFAULT_CONFIG) -> list[str]:
    """Names of the columns `build_graph_features` returns, in catalogue order."""
    return [spec.name for spec in catalogue(cfg) if spec.name in cfg.features]


# --- the static graph, for the artifact ---------------------------------------------------------


@train_only
def fit_hub_values(frame: pd.DataFrame, columns: Sequence[str], share: float) -> dict[str, Any]:
    """The values of each column that at least `share` of the rows carry, measured on train.

    A fitted object, so it goes through the guard: which values are hubs is learned from the
    training window and applied to every split. Used by the artifact's hub sweep and by the
    comparison ADR 0024 quantifies, and by nothing in the default feature path. A value that a
    large share of transactions carry joins strangers, and the sweep measures how much of the
    graph's structure is those values.
    """
    if not 0.0 < share <= 1.0:
        raise ValueError(f"share must lie in (0, 1], got {share}")
    values: dict[str, list[Any]] = {}
    report: dict[str, dict[str, Any]] = {}
    for column in columns:
        counts = frame[column].value_counts(dropna=True)
        hub = counts[counts / len(frame) >= share]
        values[column] = [
            item.item() if isinstance(item, np.generic) else item for item in hub.index.tolist()
        ]
        report[column] = {
            "n_hubs": int(hub.size),
            "n_values": int(counts.size),
            "share_rows_through_hubs": float(hub.sum() / len(frame)),
        }
    return {"share": float(share), "n_train_rows": len(frame), "values": values, "report": report}


def static_graph(
    frame: pd.DataFrame,
    cfg: GraphConfig = DEFAULT_CONFIG,
    hubs: Mapping[str, Any] | None = None,
) -> IncrementalGraph:
    """The graph over every row of `frame`, built through the same incremental code path.

    For the end-of-split summary. The caller decides which rows go in; the artifact passes the
    train split after `assert_train_only`. With `hubs` from `fit_hub_values`, those values link
    nothing. The link columns need not include card1 here, since nothing is read per entity.
    """
    stream = make_graph_stream(frame, cfg, hubs, require_anchor=False)
    graph = IncrementalGraph(stream, cfg, None)
    for start, stop in _time_blocks(stream.ts):
        graph.add_block(start, stop)
    graph.hub_report = dict(hubs["report"]) if hubs is not None else {}
    return graph


def day_of(frame: pd.DataFrame) -> npt.NDArray[np.int64]:
    """The day index the lag counts in, from stage 3's own function."""
    return transforms.day_index(frame).to_numpy(dtype="int64")


def link_column_profile(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """Cardinality, null rate and the heaviest value of each link column, on the rows given."""
    out: dict[str, Any] = {}
    for column in columns:
        series = frame[column]
        counts = series.value_counts(dropna=True)
        top_share = float(counts.iloc[0] / len(series)) if len(counts) and len(series) else None
        out[column] = {
            "n_values": int(series.nunique()),
            "null_rate": float(series.isna().mean()) if len(series) else None,
            "top_value": None if counts.empty else str(counts.index[0]),
            "top_value_share_of_rows": top_share,
        }
    return out


def spread(values: npt.NDArray[np.int64]) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
    }


def component_summary(graph: IncrementalGraph) -> dict[str, Any]:
    """The component-size distribution of a static graph, in transactions and in nodes."""
    sizes = graph.component_sizes()
    nodes = graph.component_node_counts()
    n_tx = int(sizes.sum())
    return {
        "n_transactions": n_tx,
        "n_components": int(sizes.size),
        "n_transactions_in_a_component_of_size_1": int((sizes == 1).sum()),
        "share_transactions_in_a_component_of_size_1": float((sizes == 1).sum() / n_tx)
        if n_tx
        else None,
        "giant_component": {
            "n_transactions": int(sizes[0]) if sizes.size else 0,
            "share_of_transactions": float(sizes[0] / n_tx) if n_tx else None,
            "n_nodes": int(nodes[0]) if nodes.size else 0,
        },
        "largest_ten_in_transactions": sizes[:10].tolist(),
        "size_in_transactions": spread(sizes),
        "size_in_nodes": spread(nodes),
        "n_value_nodes_present": int(
            sum(1 for node in range(graph.tx_base) if graph.uf.present[node])
        ),
        "degree_by_column": {
            column: spread(graph.value_degrees(column)) for column in graph.cfg.link_columns
        },
    }


def not_reachable() -> Mapping[str, str]:
    """What this module does not offer, so a reader finds the answer here."""
    return {
        "full-dataset components": (
            "no function here computes connected components over a whole frame and hands them "
            "back per row. `static_graph` builds the same incremental structure over whatever "
            "rows it is given and is used on the train split for the artifact; the transductive "
            "computation ADR 0024 quantifies lives in scripts/graph.py as a measurement of a "
            "wrong construction and is not importable from the package"
        ),
        "a label without a lag": (
            "`label_source` requires lag_days and `build_graph_features` reads labels through "
            "it and through nothing else"
        ),
    }
