"""Opt-in, streaming diagnostics for first maturity and separate refresh events."""
import gzip
import hashlib
import heapq
import json
import time

import torch
import torch.nn.functional as F


class CapabilityRecorder:
    def __init__(self, directory, args, experts, origins):
        self.directory = directory
        self.metadata = dict(dataset=args.data, horizon=args.pred_len, seed=args.seed)
        self.origins = origins
        self.threshold = args.alignment_threshold
        self.responsibility_threshold = args.responsibility_threshold
        self.top_k = max(1, min(args.credit_top_k, len(experts)))
        self.sample_limit = 10000
        self.sample = []
        self.seconds = 0.0
        self.count = 0
        self.refresh_events = 0
        self.reassessments = 0
        self.max_representation_bytes = 0
        self.full_rep = getattr(args, "rebuttal_full_rep_proxy", False)
        self.slices = []
        limit = int(getattr(args, "rebuttal_parameter_proxy_size", 0))
        if not 0 <= limit <= 256:
            raise ValueError("parameter proxy size must be in [0, 256]")
        for expert in experts:
            params = list(expert.parameters())
            total = sum(p.numel() for p in params)
            n = min(limit, total)
            positions = [(i * (total-1)) // max(1, n-1) for i in range(n)]
            offset = 0
            pieces = []
            for param in params:
                indices = [i-offset for i in positions if offset <= i < offset+param.numel()]
                if indices:
                    pieces.append((param, torch.tensor(indices, device=param.device)))
                offset += param.numel()
            self.slices.append(pieces)
        self.parameter_snapshot_bytes = 4 * sum(len(indices) for expert in self.slices for _, indices in expert)

    def __enter__(self):
        self.file = gzip.open(self.directory / "capability_unique.jsonl.gz", "wt", encoding="utf-8")
        self.refresh_file = gzip.open(self.directory / "capability_refresh_events.jsonl.gz", "wt", encoding="utf-8")
        return self

    def __exit__(self, exc_type, *_):
        self.file.close()
        self.refresh_file.close()
        if exc_type is None:
            with (self.directory / "capability_sample.json").open("w", encoding="utf-8") as f:
                json.dump({"source": "new stream; deterministic hash sample of eligible high-alignment owners",
                           "sample_keys": [[origin, expert] for _, origin, expert in sorted(self.sample, reverse=True)],
                           "sample_limit": self.sample_limit, "unique_owner_rows": self.count,
                           "refresh_events": self.refresh_events,
                           "reassessments": self.reassessments,
                           "parameter_snapshot_bytes_per_pending_record": self.parameter_snapshot_bytes,
                           "max_representation_bytes_per_pending_record": self.max_representation_bytes,
                           "diagnostic_seconds": self.seconds}, f)

    def snapshot(self, representations):
        start = time.perf_counter()
        samples = [torch.cat([p.detach().reshape(-1).index_select(0, idx) for p, idx in expert]).float().cpu()
                   if expert else None for expert in self.slices]
        rep = representations[0].detach().float().cpu() if self.full_rep else None
        if rep is not None:
            self.max_representation_bytes = max(self.max_representation_bytes, rep.numel() * rep.element_size())
        self.seconds += time.perf_counter()-start
        return samples, rep

    def compare(self, record, representations, old_predictions, new_predictions):
        start = time.perf_counter()
        current, rep = self.snapshot(representations)
        previous, old_rep = record.metadata["rebuttal_proxy_snapshot"]
        output = (new_predictions-old_predictions).square().mean(dim=(0, 1)).sqrt().detach().cpu().tolist()
        parameters = [(a-b).square().mean().sqrt().item() if a is not None else None
                      for a, b in zip(current, previous)]
        full = (F.cosine_similarity(rep, old_rep, dim=-1).tolist() if rep is not None
                else [None] * len(output))
        self.seconds += time.perf_counter()-start
        return output, parameters, full

    def maturity(self, completed, current_origin, candidates, admitted):
        start = time.perf_counter()
        d = completed.metadata["version_diagnostic"]
        chosen = set(completed.sample_responsibility.topk(self.top_k).indices.tolist())
        candidate_by_expert = {item.expert_id: item for item in candidates}
        for i, responsibility in enumerate(d["prediction_responsibility"]):
            eligible = i in chosen and responsibility >= self.responsibility_threshold
            alignment = d["capability_alignment"][i]
            high = alignment >= self.threshold
            if eligible and high:
                key = (*self.metadata.values(), completed.origin, i)
                rank = int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "big")
                entry = (-rank, completed.origin, i)
                if len(self.sample) < self.sample_limit:
                    heapq.heappush(self.sample, entry)
                elif entry > self.sample[0]:
                    heapq.heapreplace(self.sample, entry)
            item = candidate_by_expert.get(i)
            row = self.metadata | {
                "event": "first_maturity", "origin": completed.origin, "maturity_origin": current_origin,
                "expert_id": i, "stream_position": completed.origin / max(1, self.origins),
                "prediction_expert_version": d["global_expert_update_count"]-d["expert_update_delta"],
                "current_expert_version": d["global_expert_update_count"],
                "prediction_time_loss": d["prediction_expert_mse"][i],
                "current_loss": d["current_expert_mse"][i],
                "relative_degradation": d["relative_capability_degradation"][i],
                "degraded_by_gate_margin": d["capability_degraded"][i],
                "alignment": alignment, "high_alignment": high,
                "category": d["capability_evolution"][i],
                "responsibility": responsibility, "top_k": i in chosen,
                "responsibility_pass": responsibility >= self.responsibility_threshold,
                "eligible": eligible, "candidate": item is not None,
                "admitted": admitted.get(i), "sample_index": "capability_sample.json",
                "output_rmse": d["rebuttal_proxies"][0][i],
                "parameter_sample_rms_distance": d["rebuttal_proxies"][1][i],
                "full_encoder_cosine": d["rebuttal_proxies"][2][i],
            }
            self.file.write(json.dumps(row, allow_nan=False) + "\n")
            self.count += 1
        self.seconds += time.perf_counter()-start

    def reassessment(self, origin, item, memory_kind, alignment, loss):
        start = time.perf_counter()
        row = self.metadata | {
            "event": "owner_reassessment", "origin": origin, "historical_origin": item.origin,
            "expert_id": item.expert_id, "memory_kind_before": memory_kind,
            "reference_loss": item.get_reference_capability_loss(),
            "current_loss": loss, "alignment": alignment,
        }
        self.refresh_file.write(json.dumps(row, allow_nan=False) + "\n")
        self.reassessments += 1
        self.seconds += time.perf_counter()-start

    def refresh(self, origin, stats):
        start = time.perf_counter()
        self.refresh_file.write(json.dumps(self.metadata | {"event": "refresh_tick", "origin": origin, **stats}) + "\n")
        self.refresh_events += 1
        self.seconds += time.perf_counter()-start
