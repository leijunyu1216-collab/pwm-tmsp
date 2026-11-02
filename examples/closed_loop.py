"""Small installed-package example: fixed-cycle feedback and JSON restart."""
import json
from tmsp.closed_loop import calibration_graph

graph=calibration_graph('L2',2000)
for _ in range(200):graph.step()
snapshot=json.loads(json.dumps(graph.snapshot()))
for _ in range(100):last=graph.step()['observer']['record']
resumed=calibration_graph('L2',2000);resumed.restore(snapshot)
for _ in range(100):replayed=resumed.step()['observer']['record']
assert last==replayed
print(dict(cycle=last[0],target_fs=last[3],command_fs=last[4],model_width_fs=last[10]))
