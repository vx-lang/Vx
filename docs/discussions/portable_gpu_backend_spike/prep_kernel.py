# Spike-only rewrite of a structured Vx kernel into the form a buffer-binding target needs:
#  1. each memref<?x?xT> parameter becomes a flat memref<?xT>, and its row length becomes a new
#     index parameter at the end of the list (what the NVIDIA path passes as sizes/strides);
#  2. a load or store through a unit-stride 1-D reinterpret_cast view becomes a load or store on
#     the flat parameter at offset + index.
# Lines are handled in order, so a name that sibling regions reuse resolves to its latest definition.
import re, sys

src = open(sys.argv[1]).read().split('\n')
out = []
lens = {}      # parameter -> row-length parameter, for the current kernel
views = {}     # view -> (flat source, offset)
consts1 = set()
n = 0
for line in src:
    m = re.match(r'(\s*)gpu\.func (@\w+)\((.*?)\) kernel(.*)$', line)
    if m:
        ind, name, params, rest = m.groups()
        lens, views, consts1 = {}, {}, set()
        new, extra = [], []
        for p in params.split(', '):
            pname, ptype = [x.strip() for x in p.split(':', 1)]
            mt = re.match(r'memref<\?x\?x(\w+)>$', ptype)
            if mt:
                new.append(f'{pname}: memref<?x{mt.group(1)}>')
                lens[pname] = f'%vx_len_{pname[1:]}'
                extra.append(f'{lens[pname]}: index')
            else:
                new.append(p)
        out.append(f'{ind}gpu.func {name}({", ".join(new + extra)}) kernel{rest}')
        out.append(f'{ind}  %vx_c0 = arith.constant 0 : index')
        continue
    mc = re.match(r'\s*(%\w+) = arith\.constant 1 : index', line)
    if mc:
        consts1.add(mc.group(1))
    md = re.match(r'(\s*)(%\w+) = memref\.dim (%\w+), (%\w+) : memref<\?x\?x\w+>', line)
    if md:
        ind, res, arg, d = md.groups()
        assert d in consts1 and arg in lens, line
        out.append(f'{ind}{res} = arith.addi {lens[arg]}, %vx_c0 : index')
        continue
    mv = re.match(r'\s*(%\w+) = memref\.reinterpret_cast (%\w+) to offset: \[(%\w+)\], sizes: \[%\w+\], strides: \[1\] : memref<\?x\?x(\w+)> to memref<\?x\w+, strided<\[1\], offset: \?>>', line)
    if mv:
        views[mv.group(1)] = (mv.group(2), mv.group(3), mv.group(4))
        continue
    ml = re.match(r'(\s*)(%\w+) = memref\.load (%\w+)\[(%\w+)\] : memref<\?x\w+, strided<\[1\], offset: \?>>', line)
    if ml:
        ind, res, view, i = ml.groups()
        s, off, t = views[view]
        n += 1
        out.append(f'{ind}%vx_i{n} = arith.addi {off}, {i} : index')
        out.append(f'{ind}{res} = memref.load {s}[%vx_i{n}] : memref<?x{t}>')
        continue
    ms = re.match(r'(\s*)memref\.store (%\w+), (%\w+)\[(%\w+)\] : memref<\?x\w+, strided<\[1\], offset: \?>>', line)
    if ms:
        ind, val, view, i = ms.groups()
        s, off, t = views[view]
        n += 1
        out.append(f'{ind}%vx_i{n} = arith.addi {off}, {i} : index')
        out.append(f'{ind}memref.store {val}, {s}[%vx_i{n}] : memref<?x{t}>')
        continue
    assert 'reinterpret_cast' not in line and 'memref.dim' not in line, line
    out.append(line)
open(sys.argv[2], 'w').write('\n'.join(out))
