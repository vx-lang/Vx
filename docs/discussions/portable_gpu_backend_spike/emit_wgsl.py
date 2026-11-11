# Spike prototype: the WGSL twin of emit_msl.py. It reuses the same walk over the structured
# kernel and changes only how each piece is spelled. Not validated: no Naga or Tint on this machine.
import re, sys
import importlib.util
spec = importlib.util.spec_from_file_location('m', sys.argv[0].replace('emit_wgsl.py', 'emit_msl.py'))
src = open(spec.origin).read().replace('\nmain()\n', '\n')
m = {}
exec(compile(src, spec.origin, 'exec'), m)
m['TYPES'].update({'i32': 'i32', 'index': 'i32', 'f32': 'f32', 'i1': 'bool', 'i64': 'i64', 'f16': 'f16'})
m['BUILTIN'].update({'block_id': 'vx_wg', 'thread_id': 'vx_lid', 'grid_dim': 'vx_nwg', 'block_dim': 'vx_wgsize'})

def to_wgsl(line):
    # C-style declarations to WGSL ones, line by line.
    line = re.sub(r'^(\s*)const (\w+) (\w+) = (.*);', r'\1const \3: \2 = \4;', line)
    line = re.sub(r'^(\s*)(int|float|bool|i32|f32) (\w+) = \((\w+)\)(.*);', r'\1var \3: \2 = \2(\5);', line)
    line = re.sub(r'^(\s*)(int|float|bool|i32|f32) (\w+) = (.*);', r'\1var \3: \2 = \4;', line)
    line = re.sub(r'^(\s*)(int|float|bool|i32|f32) (\w+);', r'\1var \3: \2;', line)
    line = re.sub(r'(\w+) \? (\w+) : (\w+)', r'select(\3, \2, \1)', line)
    line = line.replace('while (true)', 'loop').replace('threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);', 'storageBarrier(); workgroupBarrier();')
    line = re.sub(r'\bint\b', 'i32', line); line = re.sub(r'\bfloat\b', 'f32', line)
    line = re.sub(r'(\d+\.\d+)f\b', r'\1', line)
    line = re.sub(r'if \((.*)\) \{', r'if (\1) {', line)
    return line

kernel_msl = m['kernel']
def kernel(header, body):
    text = kernel_msl(header, body).split('\n')
    sig_end = next(i for i, l in enumerate(text) if l.rstrip().endswith(') {'))
    mm = re.match(r'gpu\.func @(\w+)\((.*?)\) kernel', header)
    name, params = mm.group(1), mm.group(2)
    decls, scalars = [], []
    for k, p in enumerate(params.split(', ')):
        pname, ptype = [x.strip() for x in p.split(':', 1)]
        mt = re.match(r'memref<\?x(\w+)>', ptype)
        if mt:
            decls.append(f'@group(0) @binding({k}) var<storage, read_write> {m["v"](pname)}: array<{m["TYPES"][mt.group(1)]}>;')
        else:
            scalars.append(f'  {m["v"](pname)}: {m["TYPES"][ptype]},')
    decls.append(f'struct {name}_args {{\n' + '\n'.join(scalars) + '\n}')
    decls.append(f'@group(1) @binding(0) var<uniform> {name}_a: {name}_args;')
    body_lines = [to_wgsl(l) for l in text[sig_end + 1:]]
    for s in scalars:
        f = s.strip().split(':')[0]
        body_lines = [re.sub(r'\b' + f + r'\b', f'{name}_a.{f}', l) for l in body_lines]
    entry = (f'@compute @workgroup_size(256)\nfn {name}(@builtin(workgroup_id) vx_wg: vec3<u32>, '
             f'@builtin(local_invocation_id) vx_lid: vec3<u32>, @builtin(num_workgroups) vx_nwg: vec3<u32>) {{\n'
             f'  let vx_wgsize = vec3<u32>(256u, 1u, 1u);')
    return '\n'.join(decls) + '\n' + entry + '\n' + '\n'.join(body_lines)
m['kernel'] = kernel
lines = open(sys.argv[1]).read().split('\n')
pieces = []
for i, l in enumerate(lines):
    if l.strip().startswith('gpu.func'):
        pieces.append(kernel(l.strip(), lines[i + 1:]))
open(sys.argv[2], 'w').write('\n\n'.join(pieces) + '\n')
