"""Audit actual Qwen3-8B compiler dumps, excluding const-eval/setup functions.

Usage: python audit_fusions.py TP1_EXPORT_DIR TP2_EXPORT_DIR
The optional --check asserts the expected batch-one fusion coverage and
verifies every TP2 residual add is downstream of an all-reduce.
"""
import argparse
from collections import Counter
import json
import gzip
from pathlib import Path
import re

OP = re.compile(r'^\s*(?:(.*?) = )?"(ttnn\.\w+)"\(([^)]*)\)(.*)$')
FUNC = re.compile(r'^\s*func\.func\s+(?:private|public)\s+@([\w$]+)\(', re.M)
VALUE = re.compile(r'%[\w]+(?:#[0-9]+)?')


def inspect(path):
    text = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
    functions = list(FUNC.finditer(text))
    reports = []
    for index, function in enumerate(functions):
        if not function[1].startswith('trace_'):
            continue
        end = functions[index + 1].start() if index + 1 < len(functions) else len(text)
        body = text[function.end():end]
        ops = []
        producers = {}
        for line in body.splitlines():
            match = OP.match(line)
            if not match:
                continue
            results = VALUE.findall(match[1] or '')
            op = {'name': match[2], 'inputs': VALUE.findall(match[3]), 'line': line.strip()}
            ops.append(op)
            for result in results:
                producers[result] = op
        counts = Counter(op['name'] for op in ops)
        if counts['ttnn.matmul'] + counts['ttnn.linear'] < 100:
            continue
        is_decode = counts['ttnn.paged_scaled_dot_product_attention_decode'] != 0
        swiglu = [op for op in ops if 'ttnn.fused_swiglu' in op['line']]
        batch = None
        if swiglu:
            batch = int(re.search(r': \(tensor<(\d+)x', swiglu[0]['line'])[1])

        def through_reshape(value):
            node = producers.get(value)
            while node and node['name'] == 'ttnn.reshape':
                node = producers.get(node['inputs'][0])
            return node

        residual_chains = []
        qkv_projections = []
        for op in ops:
            if op['name'] == 'ttnn.add':
                for operand in op['inputs']:
                    reduction = through_reshape(operand)
                    if reduction and reduction['name'] == 'ttnn.all_reduce':
                        projection = through_reshape(reduction['inputs'][0])
                        if projection and projection['name'] == 'ttnn.matmul':
                            residual_chains.append([projection['line'], reduction['line'], op['line']])
            if op['name'] == 'ttnn.nlp_create_qkv_heads_decode':
                projection = through_reshape(op['inputs'][0])
                if projection and projection['name'] == 'ttnn.matmul':
                    qkv_projections.append(projection['line'])
        rms_configs = Counter(re.search(r'compute_config = (.*?), epsilon', op['line'])[1]
                              for op in ops if op['name'] == 'ttnn.rms_norm')
        weight_shapes = Counter()
        for op in ops:
            if op['name'] in ('ttnn.matmul', 'ttnn.linear'):
                shape = re.search(r'tensor<(\d+)x(\d+)x!ttcore.tile<32x32, bfp_bf8>', op['line'])
                if shape:
                    weight_shapes[f'{shape[1]}x{shape[2]}'] += 1
        reports.append({
            'file': str(path), 'function': function[1],
            'phase': 'decode' if is_decode else 'prefill', 'batch': batch,
            'counts': dict(sorted(counts.items())), 'fused_swiglu': len(swiglu),
            'qkv_projection_chains': len(qkv_projections),
            'bf8_projection_weight_shapes': dict(sorted(weight_shapes.items())),
            'matmul_transpose_b': sum(op['name'] == 'ttnn.matmul' and 'transpose_b = true' in op['line'] for op in ops),
            'residual_after_all_reduce': len(residual_chains),
            'rms_compute_configs': dict(rms_configs),
            'examples': {'swiglu': swiglu[0]['line'] if swiglu else None,
                         'qkv_projection': qkv_projections[:1],
                         'residual_chain': residual_chains[:1],
                         'linear': next((op['line'] for op in ops if op['name'] == 'ttnn.linear'), None)},
        })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tp1', type=Path)
    parser.add_argument('tp2', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = {}
    for tp, root in ((1, args.tp1), (2, args.tp2)):
        records = []
        for path in sorted((root / 'irs').glob('ttnn_*.mlir*')):
            if not path.name.startswith('ttnn_runtime_'):
                records.extend(inspect(path))
        result[f'tp{tp}'] = records
        if args.check:
            decode = [r for r in records if r['phase'] == 'decode' and r['batch'] == 1]
            assert decode, f'TP{tp}: no batch-one decode graph found'
            expected_weights = {f'4096x{6144 // tp}': 36,
                                f'4096x{24576 // tp}': 36,
                                f'{4096 // tp}x4096': 36,
                                f'{12288 // tp}x4096': 36,
                                f'{151936 // tp}x4096': 1}
            for record in records:
                assert record['bf8_projection_weight_shapes'] == expected_weights
                assert record['matmul_transpose_b'] == 1

            for record in decode:
                count = record['counts']
                assert record['fused_swiglu'] == 36
                assert record['qkv_projection_chains'] == 36
                assert count['ttnn.rms_norm'] == 145
                assert count['ttnn.rotary_embedding'] == 72
                assert count.get('ttnn.silu', 0) == 0
                assert count['ttnn.matmul'] + count.get('ttnn.linear', 0) == 145
                assert count.get('ttnn.linear', 0) == (72 if tp == 1 else 0)
                assert record['residual_after_all_reduce'] == (72 if tp == 2 else 0)
    if args.check:
        portable = ('ttnn.rms_norm', 'ttnn.rotary_embedding',
                    'ttnn.nlp_create_qkv_heads_decode', 'ttnn.silu')
        for phase, batch in (('prefill', None), ('decode', 1), ('decode', 2)):
            groups = [[r for r in result[f'tp{tp}']
                       if r['phase'] == phase and r['batch'] == batch] for tp in (1, 2)]
            assert all(groups), f'Missing graph for {phase}, batch={batch}'
            reference = groups[0][0]
            for record in groups[0] + groups[1]:
                assert record['fused_swiglu'] == reference['fused_swiglu']
                assert record['qkv_projection_chains'] == reference['qkv_projection_chains']
                assert record['rms_compute_configs'] == reference['rms_compute_configs']
                for name in portable:
                    assert record['counts'].get(name, 0) == reference['counts'].get(name, 0)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
