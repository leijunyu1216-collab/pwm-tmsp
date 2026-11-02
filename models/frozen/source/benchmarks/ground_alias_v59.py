"""Exact elimination of the zero-volt ground-sense source,not a tolerance change.

Vss vss 0 0 imposes vss=0;its branch current only closes the otherwise unused
ground KCL row. Neither variable stores charge. Observed/controlled source
current is refused because that interface would need a separate reconstruction.
"""
import hashlib,re


def eliminate(text):
    if re.search(r'(?i)\bi\(\s*vss\s*\)|vss#branch',text):
        raise ValueError('ground-source current is observed or used in feedback')
    lines=[];removed=[];changed=[]
    for line in text.splitlines(keepends=True):
        tokens=line.split()
        if not tokens:lines.append(line);continue
        name=tokens[0].lower()
        if name=='vss':
            if [x.lower() for x in tokens]!=['vss','vss','0','0']:
                raise ValueError('ground source is not the exact zero-volt constraint')
            removed.append(line);continue
        kind=name[0];indices=[]
        if kind=='m':indices=range(1,5)
        elif kind in ['r','c','v']:indices=range(1,3)
        elif kind=='x':
            # The flattened schematic retains only these two-terminal
            # auxiliary MIM wrappers;other hierarchical forms fail closed.
            if any(x.lower()=='vss' for x in tokens):
                if len(tokens)<4 or tokens[3].lower() not in ['physical_aux','sky130_fd_pr__cap_mim_m3_1']:
                    raise ValueError('unsupported hierarchical ground connection')
                indices=range(1,3)
        replaced=[i for i in indices if i<len(tokens) and tokens[i].lower()=='vss']
        if replaced:
            changed.append(dict(instance=tokens[0],node_positions=replaced))
            for i in replaced:tokens[i]='0'
            line=' '.join(tokens)+('\n' if line.endswith('\n') else '')
        lines.append(line)
    if len(removed)!=1 or not changed:raise ValueError('expected one global zero-volt ground source')
    result=''.join(lines)
    return result,dict(operation='exact ground-alias/KCL-variable elimination',eliminated_algebraic_unknowns=2,
        removed_source=removed[0].strip(),changed_components=len(changed),changed_node_connections=sum(len(x['node_positions']) for x in changed),
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),reduced_sha256=hashlib.sha256(result.encode()).hexdigest(),
        device_parameters_and_clock_inputs_changed=False,stored_dynamic_states_removed=False,
        tolerance_or_solver_method_changed=False,source_current_not_observed=True)
