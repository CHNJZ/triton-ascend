
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_loc_coverage.py v3 — 批量 loc 完整性扫描(作用域按 IR kernel 名定位 + host 过滤 + 折叠扩展)

用法:
    python3 check_loc_coverage.py <root_dir> [--out analysis.log] [--src 源码] [--max-lines 1000] [--verbose]

<root_dir> 下含多个子目录, 每个子目录是一次 dump(内有 kernel.ttir.mlir,
可选 kernel.ttadapter.mlir / kernel.adapter.mlir)。
关键修复(相对 v2):
  - 作用域: 用 IR 里真实 kernel 名定位源码函数, 名字匹配失败再退化到"最内层包含函数",
    彻底避免 ast.walk BFS 命中外层 test 包装函数(test_xxx)导致把 host 变量当 kernel 变量;
  - host 过滤: 跳过 RHS 为 torch./test_common./np./.npu()/None/字符串 的赋值(非 device 张量);
  - 折叠扩展: 索引/偏移算术(广播 [:,None]/[None,:]、//、%、index 名+算术)、block_ptr/tensor_ptr/
    descriptor、SSA 别名、常量 fold 各自归入已知大类, 不再散成几百条未知。
"""
import argparse, ast, os, re, sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ===================== loc 解析(递归穿透 name/callsite/fused) =====================
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

# ===================== 源码侧(作用域按 kernel 名定位 + host 过滤) =====================
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

def find_kernel_fn(tree, func_line, kernel_name):
    """名字优先, 退化到最内层包含 func_line 的函数(避免 BFS 命中外层 test 包装)。"""
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
            enc.sort(key=lambda n: n.end_lineno - n.lineno)  # 最小跨度 = 最内层
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

# ===================== 单目录分析 =====================
@dataclass
class KernelRaw:
    ir_dir: str; func: str; src_file: str; src_lines: List[str]
    span: Optional[Tuple[int, int]]; expected: Dict[str, Tuple[str, List[int]]]
    t_present: Set[str]; a_present: Optional[Set[str]]
    t_lines: Set[int]; a_lines: Optional[Set[int]]

def find_file(d, patterns):
    try: entries = sorted(os.listdir(d))
    except OSError: return None
    for p in patterns:
        for f in entries:
            if re.fullmatch(p, f): return os.path.join(d, f)
    return None

def func_name_of(ttir_path):
    text = open(ttir_path, encoding='utf-8', errors='replace').read()
    m = re.search(r'(?:tt\.func|func\.func)[^@]*@(\w+)', text)
    return m.group(1) if m else os.path.basename(os.path.dirname(ttir_path))

def analyze_one(ir_dir, src_override):
    ttir = find_file(ir_dir, [r'kernel\.ttir\.mlir', r'.*\.ttir(\.mlir)?'])
    if not ttir: return None
    adp = find_file(ir_dir, [r'kernel\.ttadapter\.mlir', r'kernel\.adapter\.mlir', r'.*adapter.*\.mlir'])
    func = func_name_of(ttir); kfile, func_line = locate_source(ttir)
    span, expected, src_lines = None, {}, []
    src = src_override or kfile
    if src and not os.path.exists(src):
        alt = os.path.join(ir_dir, os.path.basename(src)); src = alt if os.path.exists(alt) else None
    if src and func_line: span, expected, src_lines = extract_expected(src, func_line, func)
    t_names, t_lbf = collect_ir_info(ttir)
    if not expected: expected = {n: ('IR内名字', []) for n in t_names}
    a_names = a_lbf = None
    if adp: a_names, a_lbf = collect_ir_info(adp)
    kf = kfile or (max(t_lbf, key=lambda f: len(t_lbf[f])) if t_lbf else '')
    return KernelRaw(ir_dir, func, kfile or '', src_lines, span, expected, t_names,
                     a_names, t_lbf.get(kf, set()),
                     (a_lbf.get(kf, set()) if a_lbf is not None else None))

# ===================== 行/构造判定 + 已知大类折叠 =====================
def line_text(lines, n): return lines[n-1] if lines and 1 <= n <= len(lines) else ''
def code_part(text): return text.split('#', 1)[0].rstrip()
def is_interesting(text):
    c = code_part(text)
    if not c.strip(): return False
    has_assign = re.search(r'(?<![=!<>+\-*/])=(?!=)', c) is not None
    has_call = ('tl.' in c) or ('.load' in c) or ('.store' in c)
    return has_assign or has_call
def normalize(text): return ' '.join(code_part(text).split())

def _lhs(s):
    m = re.match(r'^(\w+)\s*=(?!=)', s); return m.group(1) if m else ''
def _rhs(s):
    return s.split('=', 1)[-1].strip() if ('=' in s and not s.startswith('def')) else ''
_INDEX_NAME = re.compile(r'(idx|index|offset|linear|addr|ptr|stride|cols|rows)', re.I)
def _is_index_name(name):
    return bool(name) and (bool(_INDEX_NAME.search(name)) or bool(re.fullmatch(r'[xyzrij]\d*', name))
                           or name in ('x', 'y', 'z', 'r'))

def known_class(stage, construct):
    """命中已知大类则返回类名(折叠), 否则 None(未知, 需逐条定位)。"""
    s = construct
    if not s: return None
    if s.startswith('def '): return '内联 helper / 组合函数签名'
    if 'tl.program_id' in s: return 'tl.program_id'
    if 'tl.arange' in s or 'tl.make_range' in s or 'torch.arange' in s: return 'tl.arange / make_range'
    if any(t in s for t in ('make_block_ptr', 'make_tensor_ptr', 'make_tensor_descriptor', 'tl.advance', '.advance(')):
        return 'block_ptr / tensor_ptr / descriptor'
    if 'tl.where' in s or re.search(r'[<>]=?', s) or re.search(r'\bmask\b', s, re.I):
        return 'mask 比较(xmask = idx < N)'
    rhs = _rhs(s); lhs = _lhs(s)
    broadcast = ('[None' in s) or ('None]' in s) or bool(re.search(r'\[\s*:', s)) or bool(re.search(r',\s*:\s*\]', s))
    divmod_ = ('//' in s) or bool(re.search(r'(?<!%)%(?!%)', s))
    if broadcast or divmod_ or (_is_index_name(lhs) and re.search(r'[+\-*]', rhs)):
        return 'Cat A 索引/偏移被吸收'
    if re.fullmatch(r'[A-Za-z_]\w*', rhs) or 'tl.multiple_of' in s:
        return 'SSA 别名(一值两名)'
    if re.fullmatch(r'-?\d+\.?\d*', rhs):
        return '常量初值 fold (Cat B)'
    return None

KNOWN_CONCLUSION = {
    'tl.arange / make_range': 'Cat A: make_range/splat/addi 被吸进 reinterpret_cast offset。'
                              '修 BlockPtrAnalysis/PtrAnalysis 锚点 absorbLoc。',
    'Cat A 索引/偏移被吸收': '与 arange 同因: 多维线性化/广播/取模整条索引被折进 reinterpret_cast/subview offset。'
                          '同一 absorbLoc 修复, 锚点=访问 op。',
    'mask 比较(xmask = idx < N)': 'Cat A: cmpi/andi 被 MaskAnalysis 折成访问边界, 无幸存 op。'
                                  '修 MaskAnalysis 收集 loc 吐锚点, 或 #151 式 NOP。',
    'block_ptr / tensor_ptr / descriptor': 'Cat A 变体: block/tensor ptr 创建被 OffsetAnalysis 解析吸收。'
                                           '修 TritonToUnstructure/OffsetAnalysis(parseMakeTensorPtr/parseAdvance)。',
    'tl.program_id': 'Cat A: get_program_id → 函数实参读取, 无幸存 op。NOP/annotation 占位, 优先级低。',
    'SSA 别名(一值两名)': '半良性: 一个 SSA 值两个源名(x0=xindex / tl.multiple_of)。'
                       'visit_Assign 语义本质, 非转换 bug, 低优先。',
    '常量初值 fold (Cat B)': 'Cat B: 常量初值提前 fold/CSE。listener/fold 归并覆盖, 不必关 pass。',
    '内联 helper / 组合函数签名': 'reduce/scan 组合函数形参, 内联后名字落到区块参数。优先级低(reduce in/init 已能带名)。',
}

def guess_category(s):
    kc = known_class(None, s)
    return kc or 'TBD: 需对照 lib 逐条定位'

def kernel_sig(k): return (k.func, k.src_file, k.span, tuple(sorted(k.expected.keys())))

def merge_group(group):
    rep = group[0]
    rep.t_present = set().union(*(g.t_present for g in group))
    have_adp = any(g.a_present is not None for g in group)
    rep.a_present = set().union(*(g.a_present for g in group if g.a_present is not None)) if have_adp else None
    rep.t_lines = set().union(*(g.t_lines for g in group))
    have_al = any(g.a_lines is not None for g in group)
    rep.a_lines = set().union(*(g.a_lines for g in group if g.a_lines is not None)) if have_al else None
    return rep

def kernel_losses(k):
    miss_t, miss_a, fp_for = [], [], []
    for name, (kind, ls) in sorted(k.expected.items(), key=lambda x: (x[1][1] or [0])[0]):
        if kind == 'constexpr参数': continue
        in_t = name in k.t_present
        in_a = (k.a_present is not None and name in k.a_present)
        rec = (name, kind, ls, line_text(k.src_lines, ls[0]) if ls else '')
        if not in_t: (fp_for if kind == 'for归纳变量' else miss_t).append(rec)
        elif k.a_present is not None and not in_a: miss_a.append(rec)
    ttir_uncov, adp_drop = [], []
    if k.span:
        for n in range(k.span[0]+1, k.span[1]+1):
            t = line_text(k.src_lines, n)
            if not is_interesting(t): continue
            if n not in k.t_lines: ttir_uncov.append((n, t))
            elif k.a_lines is not None and n not in k.a_lines: adp_drop.append((n, t))
    return miss_t, miss_a, fp_for, ttir_uncov, adp_drop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root_dir'); ap.add_argument('--out', default=None)
    ap.add_argument('--src', default=None); ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max-lines', type=int, default=1000, dest='max_lines',
                    help='日志最大行数(默认 1000), 超出则截断')
    a = ap.parse_args()
    ttir_dirs = []
    for dirpath, _, files in os.walk(a.root_dir):
        if any(re.fullmatch(r'kernel\.ttir\.mlir', f) or re.fullmatch(r'.*\.ttir(\.mlir)?', f) for f in files):
            ttir_dirs.append(dirpath)
    ttir_dirs = sorted(set(ttir_dirs))
    if not ttir_dirs: sys.exit(f'{a.root_dir} 下未发现任何含 ttir 的目录')
    raws = []
    for d in ttir_dirs:
        try:
            r = analyze_one(d, a.src)
            if r: raws.append(r)
        except Exception as e:
            print(f'[warn] 分析 {d} 失败: {e}', file=sys.stderr)
    groups = defaultdict(list)
    for r in raws: groups[kernel_sig(r)].append(r)
    kernels = [(merge_group(g), len(g)) for g in groups.values()]
    kernels.sort(key=lambda x: x[0].func)
    patterns = {}
    def add_pat(stage, construct, func, var, lines):
        p = patterns.setdefault((stage, construct), {'count': 0, 'ex': set()})
        p['count'] += 1; p['ex'].add((func, var, tuple(lines) if lines else ()))
    per_kernel = []
    for k, inst in kernels:
        mt, ma, fp, tu, ad = kernel_losses(k)
        per_kernel.append((k, inst, mt, ma, fp, tu, ad))
        for name, kind, ls, txt in mt:
            cons = f'<参数 {name}>' if kind == '参数' else (normalize(txt) or f'<{kind}>')
            add_pat('TTIR缺名(前端)', cons, k.func, name, ls)
        for name, kind, ls, txt in ma: add_pat('adapter缺名(转换)', normalize(txt) or f'<{kind}>', k.func, name, ls)
        for n, t in ad: add_pat('adapter丢行', normalize(t), k.func, '', [n])
    out = a.out or os.path.join(a.root_dir, 'analysis.log')
    MAX_LINES = a.max_lines
    L = []; W = L.append
    ordered = sorted(patterns.items(), key=lambda x: (-x[1]['count'], x[0][0]))
    known_buckets = {}; unknown = []
    for (stage, construct), p in ordered:
        kc = known_class(stage, construct)
        if kc:
            b = known_buckets.setdefault(kc, {'count': 0, 'constructs': set(), 'vars': set(), 'funcs': set()})
            b['count'] += p['count']; b['constructs'].add(construct)
            b['vars'].update(v for _, v, _ in p['ex'] if v)
            b['funcs'].update(f for f, _, _ in p['ex'])
        else:
            unknown.append(((stage, construct), p))
    total_pat = len(patterns)
    known_cnt = sum(b['count'] for b in known_buckets.values())
    unk_cnt = sum(p['count'] for _, p in unknown)
    W('=' * 78); W('loc 完整性批量扫描报告(压缩版)'); W(f'扫描目录: {a.root_dir}')
    W(f'dump 目录 {len(ttir_dirs)} | 去重后 kernel {len(kernels)} | 相似根因模式 {total_pat}')
    W(f'已知大类命中 {known_cnt} 次(折叠) | 未知模式 {len(unknown)} 种/{unk_cnt} 次(展开)'); W('=' * 78)
    W('\n########## 一、已知高频大类(折叠, 已定性)##########')
    for name in sorted(known_buckets, key=lambda n: -known_buckets[n]['count']):
        b = known_buckets[name]; cons = sorted(b['constructs'])
        W(f'\n[{name}]  命中 {b["count"]} 次, {len(b["funcs"])} 个 kernel, 构造变体 {len(cons)} 种')
        W(f'  代表构造: {cons[0]}' + (f'  (+{len(cons)-1} 变体)' if len(cons) > 1 else ''))
        if b['vars']:
            vs = sorted(b['vars']); W(f'  变量样例: {", ".join(vs[:12])}' + (' …' if len(vs) > 12 else ''))
        W(f'  结论: {KNOWN_CONCLUSION.get(name, "见文档分析")}')
    W('\n\n########## 二、未知/低频模式(需逐条定位)##########')
    if not unknown:
        W('  (无, 所有丢失均归入已知大类)')
    else:
        W('每条 = 同阶段同源码构造的丢失; 已剔除上面的已知大类')
        budget = max(0, MAX_LINES - len(L) - 60); per = 6
        show = min(len(unknown), max(10, budget // per))
        for (stage, construct), p in unknown[:show]:
            funcs = sorted(set(f for f, _, _ in p['ex']))
            vars_ = sorted(set(v for _, v, _ in p['ex'] if v))
            W(f'\n[{stage}]  count={p["count"]}')
            W(f'  构造  : {construct or "(无源码文本)"}')
            if vars_: W(f'  变量  : {", ".join(vars_[:12])}' + (' …' if len(vars_) > 12 else ''))
            W(f'  kernel: {", ".join(funcs[:8])}' + (' …' if len(funcs) > 8 else ''))
            W(f'  猜测  : {guess_category(construct)}')
        if show < len(unknown):
            W(f'\n  … 另有 {len(unknown) - show} 种低频未知模式未展开(--max-lines 调大可见)')
    W('\n\n########## 三、按 kernel 概览(紧凑)##########')
    W(f'{"kernel":36s}{"实例":5s}{"前端缺":7s}{"转换缺":7s}{"丢行":6s}')
    clean = 0
    for k, inst, mt, ma, fp, tu, ad in per_kernel:
        if not (mt or ma or ad): clean += 1; continue
        W(f'{k.func[:35]:36s}{inst:<5d}{len(mt):<7d}{len(ma):<7d}{len(ad):<6d}')
    if clean: W(f'(另有 {clean} 个 kernel 无任何丢失, 省略)')
    W('\n' + '=' * 78)
    W('说明: 已知大类已定性, 折叠计数即可; 重点看"二、未知/低频模式"。')
    W('前端缺名查 code_generator; 转换丢名/丢行查 lib 转换 pass。')
    W('for 归纳变量与 scf iter_arg 名字 pretty 形式不可见, 已排除(非真实丢失)。')
    W('=' * 78)
    if len(L) > MAX_LINES:
        L = L[:MAX_LINES-1] + [f'... [日志超 {MAX_LINES} 行已截断]']
    open(out, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'已写入 {out}  ({len(L)} 行)')
    print(f'dump {len(ttir_dirs)} → kernel {len(kernels)} → 模式 {total_pat} '
          f'(已知折叠 {len(known_buckets)} 类 / 未知 {len(unknown)} 种)')
    if a.verbose: print('\n'.join(L))

if __name__ == '__main__':
    main()