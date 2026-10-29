"""nd2nz_gm_to_l1: observable roundtrip, including explicit NZ storage."""
import sys
from pathlib import Path

# Keep this directory off sys.path: copy.py must not shadow stdlib copy
# when any of these category files is executed directly.
_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))
from utils.tla_test_utils import matrix, selected, get_stages
from dma.copy import l1_ir_case, l1_compile_case, l1_run_case

OP = 'nd2nz_gm_to_l1'
DEPENDENCIES = ["copy_l1_to_ub", "store_ub_to_gm"]
CONFIGS = [{
    'name': 'fp16_16x16_nz4d',
    'dtype': 'fp16',
    'route': 'nd2nz',
    'm': 16,
    'n': 16,
    'physical': True
}, {
    'name': 'fp16_16x32_nz4d',
    'dtype': 'fp16',
    'route': 'nd2nz',
    'm': 16,
    'n': 32,
    'physical': True
}, {
    'name': 'fp16_32x32_nz4d_cacheNone',
    'dtype': 'fp16',
    'route': 'nd2nz',
    'm': 32,
    'n': 32,
    'physical': True,
    'cache': None
}, {
    'name': 'fp16_32x32_nz4d_cache0',
    'dtype': 'fp16',
    'route': 'nd2nz',
    'm': 32,
    'n': 32,
    'physical': True,
    'cache': 0
}, {
    'name': 'fp16_32x32_nz4d_cache1',
    'dtype': 'fp16',
    'route': 'nd2nz',
    'm': 32,
    'n': 32,
    'physical': True,
    'cache': 1
}, {
    'name': 'fp16_32x32_nz4d_cache2',
    'dtype': 'fp16',
    'route': 'nd2nz',
    'm': 32,
    'n': 32,
    'physical': True,
    'cache': 2
}, {
    'name': 'fp16_32x32_nz4d_cache3',
    'dtype': 'fp16',
    'route': 'nd2nz',
    'm': 32,
    'n': 32,
    'physical': True,
    'cache': 3
}]


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    l1_ir_case(case, output_dir)


@matrix(CONFIGS)
def test_compile(case):
    l1_compile_case(case)


@matrix(CONFIGS)
def test_run(case):
    l1_run_case(case)


def main():
    do_ir, do_compile, do_run = get_stages()
    for case in selected(CONFIGS):
        if do_ir:
            test_ir(case)
        if do_compile:
            test_compile(case)
        if do_run:
            test_run(case)


if __name__ == "__main__":
    main()
