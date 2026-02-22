<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

# QuadNeuro Engine: A 4-Mode Matrix Multiplication Accelerator

## How it works

This chip implements a configurable matrix multiplication accelerator with four operating modes, all built around **8 shared signed 8-bit integer multipliers** that are time-multiplexed across compute passes.

### Modes

| BitCode `uio[2:1]` | Mode | Compute cycles | Output bytes |
|------------|------|---------------|--------------|
| `00` | 2×2 matmul + ReLU | 1 | 12 |
| `01` | 4×4 matmul raw | 8 | 48 |
| `10` | 4×4 matmul + ReLU | 8 | 48 |
| `11` | 4×4 tiled accumulate (C += A×B) | 8 | 48 |

### Architecture

**Number format:** Signed 8 bit int inputs (−128 to 127), signed 20 bit int outputs (20-bit two's-complement wrap).

**Hardware:** 8 shared signed 8-bit multipliers. In 2×2 mode all 4 outputs are computed simultaneously (1 cycle). In 4×4 mode the multiplier bank is reused across 8 passes (2 outputs per pass).

**ReLU:** A sign-bit check and mux on each output register. Set all negative entries to 0.

**Tiled accumulate (mode 11):** C registers persist between calls. Set `uio[3]` (accum_clear) to 1 on the first tile of a new computation. This lets you multiply matrices larger than 4×4 by decomposing them into 4×4 blocks.

### FSM

```
IDLE → LOAD → COMPUTE (1 or 8 cycles) → OUTPUT → IDLE
```

## How to test

### Pin Map

| Pin | Direction | Function |
|-----|-----------|----------|
| `ui_in[7:0]` | Input | Serial data byte in |
| `uo_out[7:0]` | Output | Serial data byte out |
| `uio[0]` | Bidir | `start` (input pulse) / `done` (output high) |
| `uio[2:1]` | Input | Mode select |
| `uio[3]` | Input | Accumulator clear (mode 11 only) |

### Protocol

**Step 1 — Start:** Assert `uio[0]` high for 1 clock cycle. Set `uio[2:1]` to desired mode. Set `uio[3]` if clearing accumulator.

**Step 2 — Load:** Present bytes on `ui_in`, one per rising clock edge.

- **2×2 (8 bytes):** `A[0][0] A[0][1] A[1][0] A[1][1]` then `B[0][0] B[0][1] B[1][0] B[1][1]`
- **4×4 (32 bytes):** A row-major `A[0][0..3] A[1][0..3] A[2][0..3] A[3][0..3]` then B row-major

**Step 3 — Compute:** Automatic (1 cycle for 2×2, 8 cycles for 4×4). No action required.

**Step 4 — Read:** `uio[0]` (done) goes high. Read bytes from `uo_out`, one per clock.

- **2×2 (12 bytes):** C row-major, 3 bytes per element
- **4×4 (48 bytes):** C row-major, 3 bytes per element

**Output encoding** (3 bytes per 20-bit result):
- Byte 0: `result[7:0]`
- Byte 1: `result[15:8]`
- Byte 2: `{4'b0, result[19:16]}`

### Tiled accumulation example

To multiply two 8×4 matrices using 4×4 tiles:

```
# Block: C[0:4][0:4] = A[0:4][0:4]@B[0:4][0:4] + A[0:4][4:8]@B[4:8][0:4]
send tile A[0:4][0:4], B[0:4][0:4]  with accum_clear=1  → C  = tile1
send tile A[0:4][4:8], B[4:8][0:4]  with accum_clear=0  → C += tile2
read result
```

## External hardware

None required. Standard TinyTapeout pin interface only.

## Conclusion

While a CPU performs matrix multiplication sequentially, this chip executes 8 multiplications in parallel per cycle, completing a 4×4 multiply in just 8 cycles at microwatt power. Because matrix multiplication dominates neural network computation, accelerating it provides a major performance advantage, while ReLU adds essential nonlinearity at almost no hardware cost. Tiled accumulation and shared multipliers allow the same compact hardware to scale efficiently to larger matrices. Together, these features implement the core operations of neural network inference faster and more efficiently than a general-purpose processor.
