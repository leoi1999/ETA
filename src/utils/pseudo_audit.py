"""Audit final metrics, candidate provenance and test isolation after a run."""
import json
import sys
from pathlib import Path

import torch


def audit(root):
    results = []
    for pool in sorted(root.glob('*_shared_pseudo_pool.pt')):
        base = pool.name.removesuffix('_shared_pseudo_pool.pt') if sys.version_info >= (3, 9) else pool.name[:-len('_shared_pseudo_pool.pt')]
        split = torch.load(root / (base + '-fusion') / 'alignment_split.pt', map_location='cpu', weights_only=False)
        payload = torch.load(pool, map_location='cpu', weights_only=False)
        assert payload['split_identity'] == split['split_identity'], pool
        valid_left = set(split['validation'][:, 0].tolist())
        valid_right = set(split['validation'][:, 1].tolist())
        test_left = set(split['test'][:, 0].tolist())
        test_right = set(split['test'][:, 1].tolist())
        for a, b in payload['pairs'].tolist():
            assert a in valid_left and b in valid_right, (pool, a, b)
            assert a not in test_left and b not in test_right, (pool, a, b)
        metrics = json.loads((root / (base + '-fusion') / 'test_metrics.json').read_text())
        assert metrics['split_identity'] == split['split_identity']
        assert metrics['test_pair_count'] == len(split['test'])
        results.append({'experiment': base, 'pseudo_pairs': len(payload['pairs']), **metrics})
    # A high production threshold can legitimately yield no pool file.
    for path in sorted(root.glob('*-fusion/test_metrics.json')):
        if not any(r['experiment'] == path.parent.name[:-len('-fusion')] for r in results):
            results.append({'experiment': path.parent.name[:-len('-fusion')], 'pseudo_pairs': 0,
                            **json.loads(path.read_text())})
    if not results:
        raise RuntimeError('No final test results found.')
    (root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
