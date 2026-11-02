# PWM-TMSP

Behavioral models and architecture-level prediction for PWM time-mode signal processing.
The release contains the primitive library, generated model cards, an exact software
snapshot from the independent-network freeze, committed predictions, and measured
event data for the continuous six-word median circuit.

## Reproduce

Requires Python 3.11 or later and NumPy. From a clean checkout:

```sh
python -m pip install .
python scripts/reproduce_frozen.py
python examples/median_network.py
```

The reproduction command verifies the SHA-256 manifest, regenerates all three
forecast sets from the frozen inputs and cards, and independently recomputes the
published metrics from 6000 measured outputs. It does not need ngspice, a PDK,
the original workspace, or a physical-output oracle. The frozen source is retained
unchanged; the runner extracts its pure prediction function to avoid workspace-only
imports. Forecast comparisons allow at most 1 fs for platform arithmetic.

## Contents

| Directory | Contents |
|---|---|
| `tmsp/` | Installable original primitive and co-simulation library |
| `models/` | Primitive cards, Verilog-A source, immutable network cards and source snapshot |
| `data/inputs/` | Original 5107-input workload, including warmup, planned targets and tail |
| `data/measurements/` | 6000 measured word outputs from the completed 1000-target scope |
| `results/predictions/` | Original prospective L0, L1 and static-cubic L2 forecasts |
| `results/summary/` | Per-model aggregate errors and physical operating scope |
| `scripts/` | Simulator-free reproduction entry point |

## Measured scope

The schematic network has six physical words, 18 banks, 14 MIN/MAX gates and a
200 ns throughput period at TT/27 C/1.8 V. The completed continuous experiment
contains 100 valid warmup outputs followed by 1000 targets across 4–196 ns input
widths. All 6000 scored word outputs meet the original 2 ns width and 0.2 ns
falling-anchor bounds. Maximum errors are 1.279118848 and 0.181489543 ns.
Width-prediction RMS is 0.355366238 ns (L0), 0.161114926 ns (L1), and
0.122206851 ns (static cubic L2), a 24.15% reduction relative to L1.

The input and prediction files include the prospective 5000-target plan. Physical
validation of all 5000 targets, the complete-system three-noise workload,
integrated system PPA/PEX and a matched system-speed comparison are incomplete.
The released network L2 is a static cubic candidate, not a qualification of every
dynamic L2 model in the library. Separate routed component blocks do not establish
an integrated-system area.

## Protocol and provenance

`models/frozen/FREEZE_RECORD.json` records original hashes and protocol history.
The 1000-target evaluator was corrected after exposure to retain native SI time
ordering; circuit parameters, original inputs, predictions and acceptance limits
were unchanged. The original MM assessment reports FF edge P99 of 1.0202% FS,
above the original 1% criterion; its separate 1.5% profile was introduced after
observation. Neither result is presented as meeting the original MM criterion.

Full analog trajectories are retained locally with their SHA-256 identifiers.
This repository supplies event-level measurements for forecast/metric reproduction.
A fresh transistor simulation additionally requires ngspice, the licensed SKY130
PDK, and the complete physical environment; it is outside the simulator-free check.

## License

Original code and generated numeric model/result data are MIT licensed. See
`LICENSE` and `NOTICE.md`. Runtime binaries, the PDK, private review materials,
and the unpublished manuscript are not included.
