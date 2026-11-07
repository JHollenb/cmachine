"""Saved-operand analysis; no model, tokenizer, or candidate fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

PRIME = 2147483647

def field_values(values):
    """Exact finite IEEE binary32 rationals mapped into F_p."""
    a = np.asarray(values, dtype='<f4')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite field operand')
    result = []
    for bits in a.view('<u4').flat:
        bits = int(bits)
        exponent = (bits >> 23) & 255
        mantissa = bits & 0x7fffff
        shift = -149
        if exponent:
            mantissa |= 1 << 23
            shift = exponent - 150
        result.append(((-1 if bits >> 31 else 1) * mantissa * pow(2, shift % (PRIME-1), PRIME)) % PRIME)
    return np.asarray(result, dtype=np.int64).reshape(a.shape)

def eliminate(matrix):
    """Int64 products fit: (p-1)^2 < 2^62, with no accumulated dot products."""
    a = np.asarray(matrix, dtype=np.int64).copy() % PRIME
    rows, columns = a.shape
    rank, determinant, pivots = 0, 1, []
    for col in range(columns):
        found = np.flatnonzero(a[rank:, col])
        if not len(found):
            continue
        row = rank + int(found[0])
        if row != rank:
            a[[rank, row]] = a[[row, rank]]
            determinant = -determinant % PRIME
        pivot = int(a[rank, col])
        determinant = determinant * pivot % PRIME
        pivots.append({'column': col, 'row_swap': row, 'value': pivot})
        a[rank, col:] = a[rank, col:] * pow(pivot, PRIME-2, PRIME) % PRIME
        if rank + 1 < rows:
            a[rank+1:, col:] = (a[rank+1:, col:] - a[rank+1:, col, None] * a[rank, None, col:] % PRIME) % PRIME
        rank += 1
        if rank == rows:
            break
    return {'prime': PRIME, 'rank': rank, 'shape': [rows, columns], 'determinant_mod_prime': determinant if rows == columns and rank == rows else 0, 'pivots': pivots}

def relative(a, b):
    return float(np.linalg.norm(a-b) / max(float(np.linalg.norm(b)), 1e-30))

def first_different(a, b):
    different = np.any(a != b, axis=tuple(range(1, a.ndim))) if a.ndim > 1 else a != b
    found = np.flatnonzero(different)
    return int(found[0]) if len(found) else None

def analyze(raw, unused_bank, driver, texts):
    raw = Path(raw)
    definitions = [('closed-compact',2,False,1),('escape-stale',2,True,1),('repair-plus',3,True,1),('repair-minus',3,True,-1),('closed-plus',8,False,1),('closed-minus',8,False,-1),('escape-plus',8,True,1),('escape-minus',8,True,-1)]
    steps = driver['steps']
    actual = [(p['name'],p['rank'],p['escape_write'],p['sign']) for p in driver['paths']]
    if actual != definitions or steps not in (1,16) or driver['dimension']!=8 or driver['width']!=4:
        raise ValueError('Path geometry differs')
    if (driver['norm_calls'],driver['matvec_calls'],driver['swiglu_calls']) != (8*steps,24*steps,8*steps):
        raise ValueError('Primitive counters differ')
    for desc in driver['paths']:
        rank = desc['rank'];compact = rank<8
        if desc['steps']!=steps or desc['compact']!=compact or desc['retained_energy']!=(1 if rank==2 else 0) or desc['normalization_width']!=8 or desc['state_bytes']!=4*(rank+compact) or desc['static_affine_bytes']!=48*rank or desc['affine_macs_per_step']!=12*rank:
            raise ValueError('Path contract differs')
    def load(name, shape):
        value = np.fromfile(raw / name, dtype='<f4')
        if value.size != int(np.prod(shape)) or not np.isfinite(value).all():
            raise ValueError('Invalid recorded operand ' + name)
        return value.reshape(shape)
    seal = json.loads((raw / 'candidate-seal.json').read_text())
    candidate_names = ('closed-compact', 'escape-stale', 'repair-plus', 'repair-minus')
    expected = {f'{name}-{suffix}.f32' for name in candidate_names for suffix in ('states', 'norms', 'gates', 'ups', 'products', 'writes', 'rho')}
    expected |= {f'{name}-{suffix}.f32' for name in ('closed', 'escape') for suffix in ('G', 'U', 'D')}
    if {r['name'] for r in seal['files']} != expected or len(seal['files']) != len(expected):
        raise ValueError('Candidate seal membership differs')
    for record in seal['files']:
        data = (raw / record['name']).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Candidate seal differs')
    g, u, d = np.zeros((4,8), '<f4'), np.zeros((4,8), '<f4'), np.zeros((8,4), '<f4')
    g[0,0], g[1,1], g[2,0], g[3,1] = .5, -.75, .25, .5
    u[0,1], u[1,0], u[2,1], u[3,0] = .5, .25, -.5, .5
    d[0,0], d[1,1] = .125, -.125
    for name in ('closed', 'escape'):
        down = d.copy()
        if name == 'escape': down[2,0] = .25
        for suffix, matrix in (('G',g), ('U',u), ('D',down)):
            if load(f'{name}-{suffix}.f32', matrix.shape).tobytes() != matrix.tobytes():
                raise ValueError('Source parameter identity differs')
    paths, observations, mechanics = {}, [], []
    for desc in driver['paths']:
        name, steps, rank = desc['name'], desc['steps'], desc['rank']
        shapes = {'states':(steps+1,rank), 'norms':(steps,8), 'gates':(steps,4), 'ups':(steps,4), 'products':(steps,4), 'writes':(steps,rank), 'rho':(steps,)}
        values = {k:load(f'{name}-{k}.f32', shape) for k,shape in shapes.items()}
        initial = np.zeros(rank, '<f4'); initial[:2] = [.75,-.5]
        if rank >= 3: initial[2] = desc['sign']
        if values['states'][0].tobytes() != initial.tobytes(): raise ValueError('Initial state differs')
        # This checks the declared executed residual update, not scientific accuracy.
        expected_next = np.add(values['states'][:-1], values['writes'], dtype=np.float32)
        if values['states'][1:].tobytes() != expected_next.tobytes(): raise ValueError('Own-state addition differs')
        down = d.copy()
        if desc['escape_write']: down[2,0] = .25
        max_norm_remainder = max_gate_remainder = max_write_remainder = 0.
        for step in range(steps):
            state = values['states'][step].astype(np.float64)
            energy = float(state@state) + desc['retained_energy']
            ideal_rho = 1/np.sqrt(energy/8+1e-6)
            max_norm_remainder = max(max_norm_remainder, abs(float(values['rho'][step])-ideal_rho))
            norm = values['norms'][step,:rank].astype(np.float64)
            max_gate_remainder = max(max_gate_remainder, float(np.max(np.abs(g[:,:rank]@norm-values['gates'][step]))), float(np.max(np.abs(u[:,:rank]@norm-values['ups'][step]))))
            max_write_remainder = max(max_write_remainder, float(np.max(np.abs(down[:rank]@values['products'][step].astype(np.float64)-values['writes'][step]))))
            if rank == 8:
                before, write, after = state[2:], values['writes'][step,2:].astype(np.float64), values['states'][step+1,2:].astype(np.float64)
                cross, square = float(2*before@write), float(write@write)
                observed = float(after@after-before@before)
                observations.append({'kind':'omitted-energy-update', 'path':name, 'step':step, 'cross_term':cross, 'write_squared':square, 'observed_delta':observed, 'finite_remainder':observed-cross-square})
        mechanics.append({'path':name, 'own_state_addition_byte_exact':True, 'max_rms_scale_vs_real_arithmetic':max_norm_remainder, 'max_affine_vs_real_arithmetic':max_gate_remainder, 'max_down_vs_real_arithmetic':max_write_remainder})
        paths[name] = values
    comparisons = []
    for candidate, reference in (('closed-compact','closed-plus'),('closed-compact','closed-minus'),('escape-stale','escape-plus'),('escape-stale','escape-minus'),('repair-plus','escape-plus'),('repair-minus','escape-minus')):
        a, b = paths[candidate], paths[reference]
        rank = a['states'].shape[1]
        row = {'candidate':candidate, 'reference':reference, 'all_retained_states_byte_exact':a['states'].tobytes()==b['states'][:,:rank].tobytes(), 'first_output_difference_update':first_different(a['states'][:,:2],b['states'][:,:2]), 'first_rho_difference_read':first_different(a['rho'],b['rho']), 'final_output_relative_error':relative(a['states'][-1,:2].astype(float),b['states'][-1,:2].astype(float)), 'max_output_absolute_error':float(np.max(np.abs(a['states'][:,:2]-b['states'][:,:2]))), 'max_rho_absolute_error':float(np.max(np.abs(a['rho']-b['rho'])))}
        for field in ('gates','ups','products','writes'):
            target = b[field][:,:rank] if field == 'writes' else b[field]
            row[field+'_byte_exact'] = a[field].tobytes() == target.tobytes()
        comparisons.append(row)
        observations.append({'kind':'quotient-comparison', **row})
    plus, minus = paths['escape-plus'], paths['escape-minus']
    escape = {'initial_code_equal':bool(np.array_equal(plus['states'][0,:2],minus['states'][0,:2])), 'initial_omitted_energy_equal':bool(np.array_equal(plus['states'][0,2:]**2,minus['states'][0,2:]**2)), 'first_retained_output_separation_update':first_different(plus['states'][:,:2],minus['states'][:,:2]), 'first_norm_separation_read':first_different(plus['rho'],minus['rho']), 'final_output_pair_distance':float(np.linalg.norm(plus['states'][-1,:2].astype(float)-minus['states'][-1,:2].astype(float))), 'one_step_omitted_energy_plus':float(plus['states'][1,2:].astype(float)@plus['states'][1,2:].astype(float)), 'one_step_omitted_energy_minus':float(minus['states'][1,2:].astype(float)@minus['states'][1,2:].astype(float))}
    rank_witness = None
    if 'qwen_minor' in driver:
        tile = load('qwen-down-minor.f32', (256,256))
        if hashlib.sha256(tile.tobytes()).hexdigest() != driver['qwen_minor']['sha256']: raise ValueError('Minor identity differs')
        rank_witness = eliminate(field_values(tile))
        observations.append({'kind':'qwen-down-rank-lower-bound', **{k:v for k,v in rank_witness.items() if k!='pivots'}, 'qwen_forwards':0})
    summary = {'closed_control_byte_exact':all(r['all_retained_states_byte_exact'] for r in comparisons[:2]), 'repair_byte_exact':all(r['all_retained_states_byte_exact'] for r in comparisons[4:]), 'escape':escape, 'qwen_down_rank_lower_bound':rank_witness['rank'] if rank_witness else None, 'native_state_bytes':32, 'closed_state_bytes':12, 'repaired_state_bytes':16, 'native_affine_macs_per_update':96, 'closed_affine_macs_per_update':24, 'repaired_affine_macs_per_update':36, 'measured_speedup':None, 'full_width_rms_still_paid':True, 'supplied_not_learned_compression':True}
    return {'summary':summary, 'comparisons':comparisons, 'mechanics':mechanics, 'escape':escape, 'rank_witness':rank_witness, 'candidate_sealed_files':len(expected), 'observations':observations, 'terminal_status':'not-requested'}
