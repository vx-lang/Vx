# Spike prototype: print Metal Shading Language from a Vx kernel after
# mem2reg + lift-cf-to-scf + flattened memref parameters. It covers only the ops
# the blur kernels use; anything else stops the script with the op's name.
import re, sys

TYPES = {'i32': 'int', 'index': 'int', 'f32': 'float', 'i1': 'bool', 'i64': 'long', 'f16': 'half'}
CMP = {'slt': '<', 'sgt': '>', 'sle': '<=', 'sge': '>=', 'eq': '==', 'ne': '!=',
       'olt': '<', 'ogt': '>', 'ole': '<=', 'oge': '>=', 'oeq': '==', 'one': '!='}
BIN = {'addi': '+', 'subi': '-', 'muli': '*', 'divsi': '/', 'remsi': '%', 'andi': '&', 'ori': '|',
       'addf': '+', 'subf': '-', 'mulf': '*', 'divf': '/'}
FUN = {'minsi': 'min', 'maxsi': 'max', 'minimumf': 'fmin', 'maximumf': 'fmax'}
BUILTIN = {'block_id': 'vx_tgid', 'thread_id': 'vx_tid', 'grid_dim': 'vx_ntg', 'block_dim': 'vx_tptg'}


def v(name):
    name = name.strip()
    if '#' in name:
        base, i = name.split('#')
        return 'v_' + base[1:] + '_' + i
    return 'v_' + name[1:]


def results(lhs):
    # "%8:2" -> [v_8_0, v_8_1]; "%2" -> [v_2]
    lhs = lhs.strip()
    if ':' in lhs:
        base, n = lhs.split(':')
        return [v(base + '#' + str(i)) for i in range(int(n))]
    return [v(lhs)]


def ctype(t):
    t = t.strip()
    if t not in TYPES:
        sys.exit('unsupported type ' + t)
    return TYPES[t]


def literal(val, t):
    if t == 'f32':
        return repr(float(val)) + 'f'
    if val in ('true', 'false'):
        return val
    return val


class Emitter:
    def __init__(self, lines):
        self.lines = lines
        self.i = 0
        self.out = []

    def emit(self, depth, text):
        self.out.append('  ' * depth + text)

    def peek(self):
        return self.lines[self.i].strip()

    def region(self, depth, on_terminator):
        """Emit lines until the closing '}' of the current region; returns the closer line."""
        while True:
            line = self.peek()
            self.i += 1
            if line.startswith('}'):
                return line
            if line.startswith('^bb0('):
                continue  # block args handled by the caller
            self.op(line, depth, on_terminator)

    def op(self, line, depth, on_terminator):
        m = re.match(r'(%[\w:#]+) = (.*)', line)
        lhs, rhs = (m.group(1), m.group(2)) if m else (None, line)
        if rhs.startswith('scf.condition(') or rhs.startswith('scf.yield'):
            on_terminator(rhs, depth)
            return
        if rhs.startswith('arith.constant'):
            mm = re.match(r'arith\.constant (\S+) : (\S+)', rhs)
            self.emit(depth, f'const {ctype(mm.group(2))} {v(lhs)} = {literal(mm.group(1), mm.group(2))};')
            return
        mm = re.match(r'arith\.index_cast (%\w+) : \S+ to (\S+)', rhs)
        if mm:
            self.emit(depth, f'{ctype(mm.group(2))} {v(lhs)} = ({ctype(mm.group(2))}){v(mm.group(1))};')
            return
        mm = re.match(r'arith\.(?:cmpi|cmpf) (\w+), (%[\w#]+), (%[\w#]+) : (\S+)', rhs)
        if mm:
            self.emit(depth, f'bool {v(lhs)} = {v(mm.group(2))} {CMP[mm.group(1)]} {v(mm.group(3))};')
            return
        mm = re.match(r'arith\.select (%[\w#]+), (%[\w#]+), (%[\w#]+) : (\S+)', rhs)
        if mm:
            self.emit(depth, f'{ctype(mm.group(4))} {v(lhs)} = {v(mm.group(1))} ? {v(mm.group(2))} : {v(mm.group(3))};')
            return
        mm = re.match(r'arith\.(\w+) (%[\w#]+), (%[\w#]+) : (\S+)', rhs)
        if mm and mm.group(1) in BIN:
            self.emit(depth, f'{ctype(mm.group(4))} {v(lhs)} = {v(mm.group(2))} {BIN[mm.group(1)]} {v(mm.group(3))};')
            return
        if mm and mm.group(1) in FUN:
            self.emit(depth, f'{ctype(mm.group(4))} {v(lhs)} = {FUN[mm.group(1)]}({v(mm.group(2))}, {v(mm.group(3))});')
            return
        mm = re.match(r'gpu\.(\w+)\s+([xyz])', rhs)
        if mm and mm.group(1) in BUILTIN:
            self.emit(depth, f'int {v(lhs)} = (int){BUILTIN[mm.group(1)]}.{mm.group(2)};')
            return
        if rhs.startswith('gpu.barrier'):
            self.emit(depth, 'threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);')
            return
        mm = re.match(r'memref\.get_global @(\w+) : memref<(\d+)x(\w+), 3>', rhs)
        if mm:
            self.emit(depth, f'threadgroup {ctype(mm.group(3))}* {v(lhs)} = {mm.group(1)};')
            return
        mm = re.match(r'memref\.load (%\w+)\[(%[\w#]+)\] : memref<(?:\?|\d+)x(\w+)(?:, 3)?>', rhs)
        if mm:
            self.emit(depth, f'{ctype(mm.group(3))} {v(lhs)} = {v(mm.group(1))}[{v(mm.group(2))}];')
            return
        mm = re.match(r'memref\.store (%[\w#]+), (%\w+)\[(%[\w#]+)\] : memref<(?:\?|\d+)x(\w+)(?:, 3)?>', rhs)
        if mm:
            self.emit(depth, f'{v(mm.group(2))}[{v(mm.group(3))}] = {v(mm.group(1))};')
            return
        if rhs.startswith('scf.while'):
            self.scf_while(lhs, rhs, depth)
            return
        if rhs.startswith('scf.if'):
            self.scf_if(lhs, rhs, depth)
            return
        if rhs.startswith('gpu.return') or rhs == 'return':
            self.emit(depth, 'return;')
            return
        sys.exit('unsupported op: ' + rhs)

    def block_args(self):
        line = self.peek()
        mm = re.match(r'\^bb0\((.*)\):', line)
        if not mm:
            return []
        self.i += 1
        return [(a.split(':')[0].strip(), a.split(':')[1].strip()) for a in mm.group(1).split(',')]

    def scf_while(self, lhs, rhs, depth):
        mm = re.match(r'scf\.while \((.*?)\) : \((.*?)\) -> \(?(.*?)\)? \{', rhs)
        inits = [a.split('=') for a in mm.group(1).split(',')]
        types = [t.strip() for t in mm.group(2).split(',')]
        res_types = [t.strip() for t in mm.group(3).split(',')] if mm.group(3).strip() else []
        res = results(lhs) if lhs else []
        tag = f'w{self.i}'
        for r, t in zip(res, res_types):
            self.emit(depth, f'{ctype(t)} {r};')
        self.emit(depth, '{')
        for k, ((_, init), t) in enumerate(zip(inits, types)):
            self.emit(depth + 1, f'{ctype(t)} {tag}_it{k} = {v(init)};')
        self.emit(depth + 1, 'while (true) {')
        self.emit(depth + 2, f'bool {tag}_go;')
        for k, t in enumerate(res_types):
            self.emit(depth + 2, f'{ctype(t)} {tag}_cv{k};')
        self.emit(depth + 2, '{')
        for k, ((name, _), t) in enumerate(zip(inits, types)):
            self.emit(depth + 3, f'{ctype(t)} {v(name)} = {tag}_it{k};')

        def before_term(text, d):
            m2 = re.match(r'scf\.condition\((%[\w#]+)\)\s*(.*?)(?: : .*)?$', text)
            self.emit(d, f'{tag}_go = {v(m2.group(1))};')
            vals = [x for x in m2.group(2).split(',') if x.strip()]
            for k, x in enumerate(vals):
                self.emit(d, f'{tag}_cv{k} = {v(x)};')
        closer = self.region(depth + 3, before_term)
        assert closer.startswith('} do {'), closer
        self.emit(depth + 2, '}')
        self.emit(depth + 2, f'if (!{tag}_go) {{')
        for k, r in enumerate(res):
            self.emit(depth + 3, f'{r} = {tag}_cv{k};')
        self.emit(depth + 3, 'break;')
        self.emit(depth + 2, '}')
        self.emit(depth + 2, '{')
        for k, (name, t) in enumerate(self.block_args()):
            self.emit(depth + 3, f'{ctype(t)} {v(name)} = {tag}_cv{k};')

        def after_term(text, d):
            m2 = re.match(r'scf\.yield\s*(.*?)(?: : .*)?$', text)
            vals = [x for x in m2.group(1).split(',') if x.strip()]
            for k, x in enumerate(vals):
                self.emit(d, f'{tag}_it{k} = {v(x)};')
        self.region(depth + 3, after_term)
        self.emit(depth + 2, '}')
        self.emit(depth + 1, '}')
        self.emit(depth, '}')

    def scf_if(self, lhs, rhs, depth):
        mm = re.match(r'scf\.if (%[\w#]+)(?: -> \(?(.*?)\)?)? \{', rhs)
        res_types = [t.strip() for t in mm.group(2).split(',')] if mm.group(2) else []
        res = results(lhs) if lhs else []
        for r, t in zip(res, res_types):
            self.emit(depth, f'{ctype(t)} {r};')

        def term(text, d):
            m2 = re.match(r'scf\.yield\s*(.*?)(?: : .*)?$', text)
            vals = [x for x in m2.group(1).split(',') if x.strip()]
            for r, x in zip(res, vals):
                self.emit(d, f'{r} = {v(x)};')
        self.emit(depth, f'if ({v(mm.group(1))}) {{')
        closer = self.region(depth + 1, term)
        if closer.startswith('} else {'):
            self.emit(depth, '} else {')
            self.region(depth + 1, term)
        self.emit(depth, '}')


GLOBALS = {}


def kernel(header, body_lines):
    mm = re.match(r'gpu\.func @(\w+)\((.*?)\) kernel', header)
    name, params = mm.group(1), mm.group(2)
    access = re.search(r'vx\.arg_access = \[(.*?)\]', header)
    access = [a.strip().strip('"') for a in access.group(1).split(',')] if access else []
    out_params = []
    for k, p in enumerate(params.split(', ')):
        pname, ptype = [x.strip() for x in p.split(':', 1)]
        mt = re.match(r'memref<\?x(\w+)>', ptype)
        if mt:
            const = 'const ' if k < len(access) and access[k] == 'readonly' else ''
            out_params.append(f'device {const}{ctype(mt.group(1))}* {v(pname)} [[buffer({k})]]')
        else:
            out_params.append(f'constant {ctype(ptype)}& {v(pname)} [[buffer({k})]]')
    out_params += ['uint3 vx_tgid [[threadgroup_position_in_grid]]',
                   'uint3 vx_tid [[thread_position_in_threadgroup]]',
                   'uint3 vx_ntg [[threadgroups_per_grid]]',
                   'uint3 vx_tptg [[threads_per_threadgroup]]']
    e = Emitter(body_lines)
    e.out.append(f'kernel void {name}(' + ',\n    '.join(out_params) + ') {')
    for g, (count, t) in GLOBALS.items():
        if g.startswith(name + '_smem'):
            e.out.append(f'  threadgroup {ctype(t)} {g}[{count}];')
    e.region(1, lambda t, d: None)
    e.out.append('}')
    return '\n'.join(e.out)


def main():
    lines = open(sys.argv[1]).read().split('\n')
    pieces = ['#include <metal_stdlib>', 'using namespace metal;', '']
    i = 0
    for line in lines:
        mg = re.match(r'\s*memref\.global "private" @(\w+) : memref<(\d+)x(\w+), 3>', line)
        if mg:
            GLOBALS[mg.group(1)] = (mg.group(2), mg.group(3))
    while i < len(lines):
        if lines[i].strip().startswith('gpu.func'):
            header = lines[i].strip()
            pieces.append(kernel(header, lines[i + 1:]))
            pieces.append('')
        i += 1
    open(sys.argv[2], 'w').write('\n'.join(pieces))


main()
