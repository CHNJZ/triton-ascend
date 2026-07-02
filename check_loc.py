#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_loc.py v3 — 单目录 loc 完整性检查(详细输出版)

用法:
    python3 check_loc.py <ir_dir> [--src 源码路径]

<ir_dir> 内有 kernel.ttir.mlir 与 kernel.ttadapter.mlir(或 kernel.adapter.mlir)。
三段:
  1. 从 TTIR 抓 kernel 源, 按 IR 里真实 kernel 名定位源码函数, 带行号打印整个 kernel;
  2. 判 TTIR 每个应命名变量的 NameLoc 是否存活(前端);
  3. 判 adapter 是否存活(转换), 丢失逐变量标出, 并给分类。
关键修复(相对旧版):
  - 作用域用 IR kernel 名定位, 失败再退化到最内层包含函数, 避免误取外层 test 包装函数;
  - 跳过 host RHS(torch./test_common./.npu()/None/字符串);
  - 丢失附带分类(make_range/索引吸收/mask/block_ptr/别名/常量/...)。
exit: 0 全通过, 1 有丢失。
"""
import argparse, ast, os, re, sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

@dataclass
class Loc:
    kind: str; file: str = ""; line: int = 0; col: int = 0; name: str = ""
    children: List["Loc"] = field(default_factory=list)

class LocParser:
    def __init__(self, text):
        self.aliases = {}
        for m in re.finditer(r'^(#loc\d*)\s*=\s*loc\((.*)\)\s*$', text, re.M):
            self.aliases[m.group(1)] = m.group(2)
        self._cache = {}
    def resolve(self, e):
        e = e.strip()
        if e in self._cache: return self._cache[e]
        v = self._parse(e); self._cache[e] = v; return v
    def _parse(self, e):
        e = e.strip()
        if e.startswith('#loc'):
            inner = self.aliases.get(e)
            return self.resolve(inner) if inner is not None else Loc('unknown')
        if e == 'unknown': return Loc('unknown')
        if e.startswith('loc(') and e.endswith(')'): return self._parse(e[4:-1])
        if e.startswith('callsite('):
            body = e[len('callsite('):-1]; depth, split = 0, -1
            for i in range(len(body)-3):
                c = body[i]
                if c in '([': depth += 1
                if c in ')]': depth -= 1
                if depth == 0 and body[i:i+4] == ' at ': split = i; break
            if split < 0: return Loc('unknown')
            return Loc('callsite', children=[self._parse(body[:split]), self._parse(body[split+4:])])
        if e.startswith('fused'):
            try: body = e[e.index('[')+1:e.rindex(']')]
            except ValueError: return Loc('unknown')
            return Loc('fused', children=[self._parse(p) for p in self._split_top(body)])
        m = re.match(r'^"((?:[^"\\]|\\.)*)"(.*)$', e, re.S)
        if m:
            s, rest = m.group(1), m.group(2).strip()
            if rest.startswith(':'):
                lm = re.match(r'^:(\d+):(\d+)$', rest)
                if lm: return Loc('file', file=s, line=int(lm.group(1)), col=int(lm.group(2)))
            if rest.startswith('(') and rest.endswith(')'):
                return Loc('name', name=s, children=[self._parse(rest[1:-1])])
            if rest == '': return Loc('name', name=s, children=[Loc('unknown')])
        return Loc('unknown')
    @staticmethod
    def _split_top(s):
        out, depth, cur, inq = [], 0, '', False
        for c in s:
            if c == '"': inq = not inq
            if not inq:
                if c in '([': depth += 1
                if c in ')]': depth -= 1
                if c == ',' and depth == 0: out.append(cur); cur = ''; continue
            cur += c
        if cur.strip(): out.append(cur)
        return out

def iter_inline_locs(text):
    for line in text.splitlines():
        if re.match(r'^#loc\d*\s*=', line): continue
        i = 0
        while True:
            j = line.find('loc(', i)
            if j < 0: break
            depth, k, inq = 0, j+3, False
            while k < len(line):
                c = line[k]
                if c == '"': inq = not inq
                if not inq:
                    if c == '(': depth += 1
                    elif c == ')':
                        depth -= 1
                        if depth == 0: break
                k += 1
            yield line[j+4:k]; i = k+1

def collect_ir_info(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    p = LocParser(text); names = set(); lines_by_file = defaultdict(set); seen = set()
    def walk(l, pending=None):
        if l.kind == 'file':
            lines_by_file[l.file].add(l.line)
            if pending: names.add(pending)
        elif l.kind == 'name':
            names.add(l.name)
            for c in l.children: walk(c, l.name)
        else:
            for c in l.children: walk(c, pending)
    for expr in set(p.aliases.values()) | set(iter_inline_locs(text)):
        if expr in seen: continue
        seen.add(expr); walk(p.resolve(expr))
    return names, lines_by_file

def locate_source(ttir_path):
    text = open(ttir_path, encoding='utf-8', errors='replace').read()
    files = re.findall(r'"((?:[^"\\]|\\.)*\.py)":(\d+):\d+', text)
    if not files: return None, None
    cnt = defaultdict(int)
    for f, _ in files: cnt[f] += 1
    kfile = max(cnt, key=cnt.get)
    func_line = min(int(l) for f, l in files if f == kfile)
    return kfile, func_line

HOST_RHS = ('torch.', 'test_common.', 'np.', 'numpy.', '.npu(', '.cuda(', '.cpu(',
            'generate_tensor', '.to(', '.item(', '.size(', '.numel(', '.stride(', '.shape')
def _is_host_rhs(seg):
    if seg is None: return False
    s = seg.strip()
    if not s: return False
    if s == 'None' or s[:1] in '"\'': return True
    return any(h in s for h in HOST_RHS)

def func_name_of(ttir_path):
    text = open(ttir_path, encoding='utf-8', errors='replace').read()
    m = re.search(r'(?:tt\.func|func\.func)[^@]*@(\w+)', text)
    return m.group(1) if m else ''

def find_kernel_fn(tree, func_line, kernel_name):
    cands = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if kernel_name:
        named = [n for n in cands if n.name == kernel_name]
        if named:
            named.sort(key=lambda n: (not (n.lineno <= (func_line or n.lineno) <= n.end_lineno),
                                      n.end_lineno - n.lineno))
            return named[0]
    if func_line:
        enc = [n for n in cands if n.lineno <= func_line <= n.end_lineno]
        if enc:
            enc.sort(key=lambda n: n.end_lineno - n.lineno)
            return enc[0]
    return None

def extract_expected(src, func_line, kernel_name):
    code = open(src, encoding='utf-8', errors='replace').read()
    try: tree = ast.parse(code)
    except SyntaxError: return None, {}, code.splitlines()
    fn = find_kernel_fn(tree, func_line, kernel_name)
    if fn is None: return None, {}, code.splitlines()
    expected = {}
    def add(name, kind, line):
        k, ls = expected.get(name, (kind, []))
        if line not in ls: ls.append(line)
        expected[name] = (k, ls)
    for a in fn.args.args:
        ann = ast.get_source_segment(code, a.annotation) if a.annotation else ''
        add(a.arg, 'constexpr参数' if ann and 'constexpr' in ann else '参数', a.lineno)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if _is_host_rhs(ast.get_source_segment(code, node.value)): continue
            add(node.targets[0].id, '赋值', node.lineno)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if _is_host_rhs(ast.get_source_segment(code, node.value)): continue
            add(node.target.id, '赋值', node.lineno)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            add(node.target.id, 'for归纳变量', node.lineno)
    return (fn.lineno, fn.end_lineno), expected, code.splitlines()

# ---- 分类(与批量版一致) ----
def _lhs(s):
    m = re.match(r'^(\w+)\s*=(?!=)', s); return m.group(1) if m else ''
def _rhs(s):
    return s.split('=', 1)[-1].strip() if ('=' in s and not s.startswith('def')) else ''
_INDEX_NAME = re.compile(r'(idx|index|offset|linear|addr|ptr|stride|cols|rows)', re.I)
def _is_index_name(name):
    return bool(name) and (bool(_INDEX_NAME.search(name)) or bool(re.fullmatch(r'[xyzrij]\d*', name))
                           or name in ('x', 'y', 'z', 'r'))
def classify(construct):
    s = construct
    if not s: return 'TBD'
    if s.startswith('def '): return '内联 helper/组合函数形参'
    if 'tl.program_id' in s: return 'Cat A: program_id→实参'
    if 'tl.arange' in s or 'tl.make_range' in s or 'torch.arange' in s: return 'Cat A: arange/make_range 吸收'
    if any(t in s for t in ('make_block_ptr', 'make_tensor_ptr', 'make_tensor_descriptor', 'tl.advance', '.advance(')):
        return 'Cat A: block_ptr/tensor_ptr 吸收(OffsetAnalysis)'
    if 'tl.where' in s or re.search(r'[<>]=?', s) or re.search(r'\bmask\b', s, re.I):
        return 'Cat A: mask 折成边界(MaskAnalysis)'
    rhs = _rhs(s); lhs = _lhs(s)
    broadcast = ('[None' in s) or ('None]' in s) or bool(re.search(r'\[\s*:', s)) or bool(re.search(r',\s*:\s*\]', s))
    divmod_ = ('//' in s) or bool(re.search(r'(?<!%)%(?!%)', s))
    if broadcast or divmod_ or (_is_index_name(lhs) and re.search(r'[+\-*]', rhs)):
        return 'Cat A: 索引/偏移表达式吸收'
    if re.fullmatch(r'[A-Za-z_]\w*', rhs) or 'tl.multiple_of' in s:
        return 'SSA 别名(一值两名, 半良性)'
    if re.fullmatch(r'-?\d+\.?\d*', rhs):
        return 'Cat B: 常量初值 fold'
    return 'TBD: 需对照 lib 定位'

def line_text(lines, n): return lines[n-1] if lines and 1 <= n <= len(lines) else ''
def code_part(text): return text.split('#', 1)[0].rstrip()
def is_interesting(text):
    c = code_part(text)
    if not c.strip(): return False
    has_assign = re.search(r'(?<![=!<>+\-*/])=(?!=)', c) is not None
    has_call = ('tl.' in c) or ('.load' in c) or ('.store' in c)
    return has_assign or has_call

def find_file(d, patterns):
    try: entries = sorted(os.listdir(d))
    except OSError: return None
    for p in patterns:
        for f in entries:
            if re.fullmatch(p, f): return os.path.join(d, f)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ir_dir'); ap.add_argument('--src', default=None)
    a = ap.parse_args()
    ttir = find_file(a.ir_dir, [r'kernel\.ttir\.mlir', r'.*\.ttir(\.mlir)?'])
    adp = find_file(a.ir_dir, [r'kernel\.ttadapter\.mlir', r'kernel\.adapter\.mlir', r'.*adapter.*\.mlir'])
    if not ttir: sys.exit(f'未找到 ttir (dir={a.ir_dir})')
    print(f'TTIR    : {ttir}')
    print(f'adapter : {adp or "(无)"}')
    func = func_name_of(ttir); kfile, func_line = locate_source(ttir)
    span, expected, src_lines = None, {}, []
    src = a.src or kfile
    if src and not os.path.exists(src):
        alt = os.path.join(a.ir_dir, os.path.basename(src)); src = alt if os.path.exists(alt) else None
    if src and func_line:
        span, expected, src_lines = extract_expected(src, func_line, func)
        print(f'kernel  : @{func}  源 {kfile} (读取 {src})')
        if span:
            print(f'\n===== kernel 源码 ({span[0]}–{span[1]} 行) =====')
            for i in range(span[0], span[1]+1):
                print(f'{i:4d} | {line_text(src_lines, i)}')
    else:
        print('!! 源文件不可用, 仅做 IR 内名字对比 (--src 可指定)')

    t_names, t_lbf = collect_ir_info(ttir)
    a_names = a_lbf = None
    if adp: a_names, a_lbf = collect_ir_info(adp)
    kf = kfile or (max(t_lbf, key=lambda f: len(t_lbf[f])) if t_lbf else '')
    if not expected:
        expected = {n: ('IR内名字', []) for n in t_names}

    miss_t, miss_a, fp_for = [], [], []
    print('\n===== 变量名覆盖 =====')
    print(f'{"变量":16s}{"类别":12s}{"定义行":14s}{"TTIR":6s}{"adapter":8s}')
    for name, (kind, ls) in sorted(expected.items(), key=lambda x: (x[1][1] or [0])[0]):
        if kind == 'constexpr参数':
            print(f'{name:16s}{kind:12s}{",".join(map(str,ls)):14s}{"跳过":6s}{"跳过":8s}'); continue
        it = name in t_names
        ia = (a_names is not None and name in a_names)
        ic = '✓' if it else '✗'
        icc = ('-' if a_names is None else ('✓' if ia else '✗'))
        print(f'{name:16s}{kind:12s}{",".join(map(str,ls)):14s}{ic:6s}{icc:8s}')
        rec = (name, kind, ls, line_text(src_lines, ls[0]) if ls else '')
        if not it: (fp_for if kind == 'for归纳变量' else miss_t).append(rec)
        elif a_names is not None and not ia: miss_a.append(rec)

    def _cls(kd, t):
        return '参数(constexpr/未用或前端未挂名)' if kd == '参数' else classify(normalize_one(t))
    if miss_t:
        print('\n[前端缺名 → 查 code_generator]')
        for n, kd, ls, t in miss_t:
            print(f'  ✗ {n}({kd}) 行{ls}: {code_part(t).strip()}')
            print(f'      分类: {_cls(kd, t)}')
    if miss_a:
        print('\n[转换丢名 → 查 lib 转换 pass]')
        for n, kd, ls, t in miss_a:
            print(f'  ✗ {n}({kd}) 行{ls}: {code_part(t).strip()}')
            print(f'      分类: {_cls(kd, t)}')
    if fp_for:
        print('\n[scf 块参数: pretty 形式不可见, 误报, 忽略]')
        for n, kd, ls, t in fp_for:
            print(f'  · {n} 行{ls}')

    if span and kf:
        rng = set(range(span[0]+1, span[1]+1))
        inter = {n for n in rng if is_interesting(line_text(src_lines, n))}
        nt = sorted(n for n in inter if n not in t_lbf.get(kf, set()))
        na = []
        if a_lbf is not None:
            na = sorted(n for n in (inter & t_lbf.get(kf, set())) if n not in a_lbf.get(kf, set()))
        print(f'\n===== 行覆盖 (有意义行) =====')
        print(f'TTIR 未覆盖: {nt or "无"}')
        print(f'adapter 相对 TTIR 丢失: {na if a_lbf is not None else "(无 adapter)"}')

    if miss_t or miss_a:
        print('\n结果: FAIL'); sys.exit(1)
    print('\n结果: PASS')

def normalize_one(text):
    return ' '.join(code_part(text).split())

if __name__ == '__main__':
    main()