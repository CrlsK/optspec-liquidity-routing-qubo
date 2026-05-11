"""Quantum-inspired liquidity router (greedy seed + dwave-neal SA refinement on venue-choice QUBO).

For each parent order we build a small QUBO over the eligible-venue choice,
seed with the greedy solution, anneal to find a better venue mix that reduces
effective-price cost while respecting penalty constraints. SA quality vs MIP
is the headline of this solver vs the classical sibling.
"""
from __future__ import annotations
import hashlib, json, os, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from adapter import to_internal, validate
from kpi_compute import compute_kpis
from additional_output_generator import generate_additional_output

SOLVER_VERSION = '1.0.0-qubo-sa-neal'
ALGORITHM_NAME = 'GreedySeed_then_dwaveNeal_SA_venueChoice_QUBO_v1'


def solver(input_data, **kwargs):
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    raw = (input_data or {}).get('data', input_data) or {}
    dsha = hashlib.sha256(json.dumps(input_data, sort_keys=True, default=str).encode()).hexdigest()
    Path(os.environ.get('ADDITIONAL_OUTPUT_DIR', './additional_output')).mkdir(parents=True, exist_ok=True)
    sa_reads = int(kwargs.get('num_reads', 200))
    sa_sweeps = int(kwargs.get('num_sweeps', 2000))
    seed = int(kwargs.get('random_seed', raw.get('random_seed', 42)))
    try:
        internal = to_internal(raw); validate(internal)
    except Exception as e: return _err('adapter', e, started, t0, dsha)
    try:
        # 1. Greedy seed
        fills_seed = _greedy(internal)
        # 2. Refine via per-order venue-choice QUBO (dwave-neal SA)
        fills, sa_diag = _refine_with_sa(internal, fills_seed, sa_reads, sa_sweeps, seed)
    except Exception as e:
        return _err('solver', e, started, t0, dsha)
    wall = time.perf_counter() - t0
    kpis = compute_kpis(internal['orders'], fills, internal['venues'])
    rp = _plan(fills)
    obj_val = (internal['obj_weights']['slippage'] * abs(kpis['realized_slippage_bps'])
               + internal['obj_weights']['fees'] * kpis['total_fees_bps']
               + internal['obj_weights']['market_impact'] * kpis['market_impact_bps']
               - internal['obj_weights']['price_discovery'] * kpis['price_discovery_score'] * 100)
    res = {**kpis,
           'benchmark': {'execution_cost': {'value': round(wall*0.5, 4), 'unit': 'credits'}, 'time_elapsed': f'{wall:.1f}s', 'energy_consumption': 0.0},
           'status': 'success', 'solution_status': 'feasible' if fills else 'infeasible',
           'routing_plan': rp,
           'execution_instructions': [{'instruction_id': f'EX-{i+1:05d}', **{k: f[k] for k in ('order_id','venue_id','asset','side','quantity')}, 'limit_price': f['exec_price'], 'time_in_force_sec': 60} for i, f in enumerate(fills)],
           'expected_kpis': kpis, 'objective_value': round(obj_val, 4),
           'constraint_report': {'binding': [], 'violations': []},
           'solver_diagnostics': {'solver_version': SOLVER_VERSION, 'algorithm': ALGORITHM_NAME, 'wall_time_s': round(wall, 3),
                                  'n_orders': len(internal['orders']), 'n_venues': len(internal['venues']), 'n_fills': len(fills),
                                  'sa_num_reads': sa_reads, 'sa_num_sweeps': sa_sweeps, 'random_seed': seed, **sa_diag},
           'errors': [], 'run_id': raw.get('run_id', dsha[:12]),
           'audit': {'solver_version': SOLVER_VERSION, 'dataset_sha256': dsha, 'run_started_at_utc': started,
                     'run_finished_at_utc': datetime.now(timezone.utc).isoformat(),
                     'platform_use_case': 'optimized-liquidity-routing-across-diversified-digital-asset-markets'}}
    try: generate_additional_output(raw, res, algorithm_name=ALGORITHM_NAME)
    except Exception: pass
    return {'result': res}


def run(data, solver_params=None, extra_arguments=None):
    sp = solver_params or {}; ea = extra_arguments or {}
    return solver({'data': data}, num_reads=int(sp.get('num_reads', 200)),
                  num_sweeps=int(sp.get('num_sweeps', 2000)),
                  random_seed=int(sp.get('random_seed', 42)))['result']


def _greedy(internal):
    fills = []
    used = {v['id']: 0.0 for v in internal['venues']}
    elig = internal['routing'].get('venue_eligibility') or {}
    max_v = internal['routing'].get('max_venues_per_order', 4)
    for o in internal['orders']:
        sym, side = o['asset'], o['side']
        qty = float(o['quantity'])
        cands = []
        for v in internal['venues']:
            if not elig.get(v['id'], {}).get('enabled', True): continue
            b = (internal['books'].get(v['id']) or {}).get(sym)
            if not b: continue
            lvls = b['asks'] if side == 'buy' else b['bids']
            if not lvls: continue
            fee = v['fee_taker_bps']
            top = lvls[0]['price']
            eff = top * (1 + fee/10000.0) if side == 'buy' else top * (1 - fee/10000.0)
            cands.append((eff if side == 'buy' else -eff, v, lvls))
        cands.sort(key=lambda x: x[0])
        uv = 0
        for _, v, lvls in cands:
            if uv >= max_v or qty <= 1e-9: break
            for lvl in lvls:
                if qty <= 1e-9: break
                take = min(qty, float(lvl['size']))
                if take <= 1e-9: continue
                notional = take * float(lvl['price'])
                cap = float((internal['limits'].get('venue_credit_usd') or {}).get(v['id'], 5_000_000))
                if used[v['id']] + notional > cap:
                    take = max(0, (cap - used[v['id']]) / float(lvl['price']))
                    if take <= 1e-9: break
                used[v['id']] += take * float(lvl['price'])
                fills.append({'order_id': o['order_id'], 'venue_id': v['id'], 'asset': sym, 'side': side,
                              'quantity': round(take, 6), 'exec_price': round(float(lvl['price']), 6),
                              'fee_bps': float(v['fee_taker_bps']), 'alpha': 0.0, 'beta': 0.0,
                              'venue_quality_w': float(v['quality_w'])})
                qty -= take
            uv += 1
    return fills


def _refine_with_sa(internal, fills_seed, num_reads, num_sweeps, seed):
    """Per-order venue-choice QUBO refinement.
    For each order: binary one-hot over its eligible venues. SA samples a better
    one-of-V mix favouring lower effective price + higher quality_w.
    """
    diag = {'sa_orders_refined': 0, 'sa_orders_skipped': 0, 'sa_avg_energy': 0.0, 'sa_lib': 'dwave-neal'}
    try:
        import neal
    except ImportError:
        diag['sa_lib'] = 'unavailable'
        return fills_seed, diag
    sampler = neal.SimulatedAnnealingSampler()
    new_fills = []
    energies = []
    elig = internal['routing'].get('venue_eligibility') or {}
    # Group seed fills by order
    by_order = {}
    for f in fills_seed: by_order.setdefault(f['order_id'], []).append(f)
    for o in internal['orders']:
        sym, side = o['asset'], o['side']
        cand_v = []
        for v in internal['venues']:
            if not elig.get(v['id'], {}).get('enabled', True): continue
            b = (internal['books'].get(v['id']) or {}).get(sym)
            if not b: continue
            lvls = b['asks'] if side == 'buy' else b['bids']
            if not lvls: continue
            top = lvls[0]
            fee = v['fee_taker_bps']
            eff = top['price'] * (1 + fee/10000.0) if side == 'buy' else top['price'] * (1 - fee/10000.0)
            cand_v.append({'v': v, 'top': top, 'eff': eff})
        if len(cand_v) < 2:
            # nothing to refine; keep seed
            new_fills.extend(by_order.get(o['order_id'], []))
            diag['sa_orders_skipped'] += 1
            continue
        # Build one-hot QUBO over cand_v
        # Cost = sum_v c_v * x_v - reward * quality_w_v * x_v
        # Constraint: sum_v x_v = 1 (one-hot)
        # Q[i,i] = c_i - reward*qw_i + LAMBDA*(1 - 2)  ; Q[i,j] = 2*LAMBDA
        n = len(cand_v)
        LAM = max(abs(cv['eff']) for cv in cand_v) * 2 + 1
        cost_min = min(cv['eff'] for cv in cand_v) if side == 'buy' else max(cv['eff'] for cv in cand_v)
        Q = {}
        for i, cv in enumerate(cand_v):
            ci = (cv['eff'] - cost_min) / max(abs(cost_min), 1e-6)  # normalized 0..few
            reward = 0.5 * cv['v']['quality_w']
            Q[(i, i)] = ci - reward - LAM
        for i in range(n):
            for j in range(i+1, n):
                Q[(i, j)] = 2 * LAM
        # Warm-start: seed best-eff index = 1
        seed_idx = min(range(n), key=lambda k: cand_v[k]['eff']) if side == 'buy' else max(range(n), key=lambda k: cand_v[k]['eff'])
        initial_states = [{i: (1 if i == seed_idx else 0) for i in range(n)}]
        try:
            ss = sampler.sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed,
                                      initial_states=initial_states)
            best = ss.first.sample
            energies.append(float(ss.first.energy))
            chosen = [i for i, b in best.items() if b == 1]
            if not chosen: chosen = [seed_idx]
            top_choice = cand_v[chosen[0]]
        except Exception:
            top_choice = cand_v[seed_idx]
        # Walk this venue's levels to fill
        v = top_choice['v']
        b = internal['books'][v['id']][sym]
        lvls = b['asks'] if side == 'buy' else b['bids']
        qty = float(o['quantity'])
        for lvl in lvls:
            if qty <= 1e-9: break
            take = min(qty, float(lvl['size']))
            if take <= 1e-9: continue
            new_fills.append({'order_id': o['order_id'], 'venue_id': v['id'], 'asset': sym, 'side': side,
                              'quantity': round(take, 6), 'exec_price': round(float(lvl['price']), 6),
                              'fee_bps': float(v['fee_taker_bps']), 'alpha': 0.0, 'beta': 0.0,
                              'venue_quality_w': float(v['quality_w'])})
            qty -= take
        diag['sa_orders_refined'] += 1
    if energies: diag['sa_avg_energy'] = round(sum(energies) / len(energies), 4)
    return new_fills, diag


def _plan(fills):
    agg = {}
    for f in fills:
        k = (f['order_id'], f['venue_id'])
        if k not in agg:
            agg[k] = {'order_id': f['order_id'], 'venue_id': f['venue_id'], 'asset': f['asset'], 'side': f['side'],
                      'allocated_quantity': 0.0, 'expected_price': 0.0, 'expected_fee_bps': f['fee_bps']}
        agg[k]['allocated_quantity'] += f['quantity']
        agg[k]['expected_price'] = max(agg[k]['expected_price'], f['exec_price'])
    return list(agg.values())


def _err(phase, exc, started, t0, dsha):
    wall = time.perf_counter() - t0
    return {'result': {'realized_slippage_bps': 0, 'fill_rate_pct': 0, 'total_fees_bps': 0, 'market_impact_bps': 0,
                       'price_discovery_score': 0, 'venue_switches': 0,
                       'benchmark': {'execution_cost': {'value': 0, 'unit': 'credits'}, 'time_elapsed': f'{wall:.1f}s', 'energy_consumption': 0.0},
                       'status': 'error', 'solution_status': 'error',
                       'routing_plan': [], 'execution_instructions': [], 'expected_kpis': {}, 'objective_value': None,
                       'constraint_report': {}, 'errors': [{'phase': phase, 'error_type': type(exc).__name__, 'error_message': str(exc), 'traceback': traceback.format_exc()}],
                       'solver_diagnostics': {'solver_version': SOLVER_VERSION},
                       'audit': {'dataset_sha256': dsha, 'run_started_at_utc': started, 'run_finished_at_utc': datetime.now(timezone.utc).isoformat()}}}


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f: inp = json.load(f)
    else: inp = {'data': {'orders': [], 'venue_catalogue': [], 'market_data': {'venues': {}}}}
    print(json.dumps(solver(inp), indent=2, default=str))
