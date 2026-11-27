"""FlexKV-backed RadixCache integration for sglang.

Two ways to select this backend at server launch:

1. ``--enable-flexkv`` (default chain in ``default_radix_cache_factory``)
2. ``--radix-cache-backend=flexkv`` (explicit registry path)

Importing this package registers the explicit name with the registry,
so the second form is available without further wiring.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.registry import register_radix_cache_backend
from sglang.srt.runtime_context import get_context, get_memory, get_serving, get_spec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlexKVDraft:
    pools: tuple[object, ...]
    needs_bigram_key: bool


def prepare_flexkv_draft(
    worker: BaseSpecWorker | None, allocator: BaseTokenToKVPoolAllocator
) -> FlexKVDraft | None:
    """Describe independent draft buffers while their owning worker is available."""
    if os.environ.get("FLEXKV_CACHE_DRAFT_KV", "0") != "1":
        if getattr(worker, "primary_draft_kv_pool", None) is not None:
            raise ValueError(
                "FlexKV host hits with independent draft KV require FLEXKV_CACHE_DRAFT_KV=1"
            )
        return None

    if worker is None:
        raise ValueError("Draft KV caching requires a speculative worker")
    spec = get_spec()
    if (
        getattr(worker, "use_compact_draft_cache", False)
        or spec.speculative_dflash_draft_kv_ring
    ):
        raise ValueError("A private draft KV ring does not use target slot IDs")

    runners = worker._draft_model_runners()
    if not runners:
        raise ValueError(
            "The speculative worker has no independent draft KV buffers to cache"
        )
    for runner in runners:
        if (
            runner.runner_ids.is_fused_draft
            or runner.token_to_kv_pool_allocator is not allocator
        ):
            raise ValueError(
                "FlexKV requires independent draft buffers sharing the target allocator"
            )

    algorithm = worker.target_worker.model_runner.spec_algorithm
    # EAGLE and STANDALONE rotate draft inputs by one token. Each additional
    # multi-layer EAGLE step rotates again, beyond the bigram key's boundary.
    if (
        algorithm.is_eagle()
        and spec.enable_multi_layer_eagle
        and worker.speculative_num_steps != 1
    ):
        raise ValueError(
            "Cascaded draft input rotations need a longer prefix key than bigram"
        )
    return FlexKVDraft(
        pools=tuple(runner.token_to_kv_pool for runner in runners),
        needs_bigram_key=algorithm.is_eagle() or algorithm.is_standalone(),
    )


def _flexkv_factory(ctx):
    """Build a :class:`FlexKVRadixCache` from a ``TreeCacheBuildContext``.

    ``TreeCacheBuildContext`` carries TP rank/size and the TP group
    coordinator, but not PP/CP. We pick those up from ``get_parallel()``;
    FlexKV needs them to fan out lookup/store decisions across the full
    TP × CP × PP topology.
    """
    if get_serving().enable_streaming_session:
        # StreamingSession can retain KV without calling the inner cache's
        # completion hooks. It cannot currently commit FlexKV restore leases.
        raise ValueError("FlexKV does not support --enable-streaming-session yet")
    if ctx.enable_hierarchical_cache:
        raise ValueError("FlexKV cannot be combined with --enable-hierarchical-cache")
    if ctx.is_hybrid_ssm:
        raise NotImplementedError("FlexKV does not support Mamba/SSM pools yet")
    params = ctx.params
    draft = getattr(ctx, "flexkv_draft", None)
    if draft is not None:
        if ctx.is_hybrid_swa:
            raise ValueError("Draft KV caching requires a full-attention pool")
        if draft.needs_bigram_key and not params.is_eagle:
            params = replace(params, is_eagle=True)
    elif params.is_eagle or os.environ.get("FLEXKV_CACHE_DRAFT_KV", "0") == "1":
        raise ValueError(
            "FlexKV speculative host hits require FLEXKV_CACHE_DRAFT_KV=1 "
            "and registered draft KV pools"
        )

    if ctx.disable_radix_cache:
        raise ValueError(
            "FlexKV requires radix caching to own restored and stored prefixes"
        )
    if ctx.is_hybrid_swa and ctx.full_tokens_per_layer == 0:
        raise NotImplementedError(
            "FlexKV requires a FULL token pool; pure SWA is unsupported"
        )

    try:
        from flexkv.integration.sglang.connector import FlexKVConnector  # noqa: F401
    except ImportError as exc:
        try:
            import flexkv  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "FlexKV is not installed. Please install the FlexKV package "
                "to use --enable-flexkv."
            ) from exc
        raise RuntimeError(
            f"FlexKV is installed but incompatible version: {exc}."
        ) from exc

    from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import (
        FlexKVRadixCache,
    )
    from sglang.srt.runtime_context import get_parallel

    # The external connector consumes a flat startup config. Hand it the
    # resolved values, including runtime overrides, rather than the raw seed.
    server_args = SimpleNamespace(**get_context().resolved_server_args_dict())
    if get_memory().flexkv_config_file and not os.environ.get("FLEXKV_CONFIG_PATH"):
        os.environ["FLEXKV_CONFIG_PATH"] = get_memory().flexkv_config_file

    # PP group is always available; attn TP / attn CP groups may share
    # the regular TP group when attn DP is off — that's fine, the
    # connector treats size-1 groups as no-ops.
    try:
        pp_group = get_parallel().pp_group
    except (RuntimeError, AssertionError):
        pp_group = None
    try:
        attn_tp_group = get_parallel().attn_tp_group
    except (RuntimeError, AssertionError):
        attn_tp_group = ctx.tp_group
    try:
        attn_cp_group = get_parallel().attn_cp_group
    except (RuntimeError, AssertionError):
        attn_cp_group = None

    # PP / CP ranks: use the group's own rank_in_group view if available;
    # fall back to 0 for single-rank dims.
    pp_rank = pp_group.rank_in_group if pp_group is not None else 0
    attn_cp_rank = attn_cp_group.rank_in_group if attn_cp_group is not None else 0
    # TpModelWorker no longer carries a ParallelState; ranks are published to
    # the runtime context before the tree cache is built.
    parallel_state = get_parallel()
    if parallel_state.attn_dp_size > 1:
        dp_rank = getattr(parallel_state, "attn_dp_rank", 0)
    else:
        dp_rank = getattr(parallel_state, "dp_rank", 0) or 0

    common_kwargs = dict(
        params=params,
        model_config=ctx.model_config,
        server_args=server_args,
        tp_rank=ctx.tp_rank,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        attn_cp_rank=attn_cp_rank,
        tp_group=ctx.tp_group,
        pp_group=pp_group,
        attn_tp_group=attn_tp_group,
        attn_cp_group=attn_cp_group,
    )

    if ctx.is_hybrid_swa:
        from sglang.srt.mem_cache.registry import create_unified_radix_cache
        from sglang.srt.mem_cache.storage.flexkv.flexkv_hybrid_radix_cache import (
            FlexKVHybridRadixCache,
        )

        inner_cache = create_unified_radix_cache(ctx)
        return FlexKVHybridRadixCache(
            inner_cache=inner_cache,
            **common_kwargs,
        )

    return FlexKVRadixCache(
        tp_size=ctx.tp_size,
        draft_pools=draft.pools if draft is not None else (),
        **common_kwargs,
    )


try:
    register_radix_cache_backend("flexkv", _flexkv_factory)
except ValueError as exc:
    # The registry refuses duplicates. Importing this package twice
    # (e.g. via both --enable-flexkv and --radix-cache-backend=flexkv)
    # is fine — log and move on.
    logger.debug("flexkv backend already registered: %s", exc)
