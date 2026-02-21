# SPDX-FileCopyrightText: 2024 Your Name
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive cocotb testbench for tt_um_matmul
# Compatible with cocotb v2.x — uses only RisingEdge, no ReadOnly polling loops

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

# ─── Golden model ─────────────────────────────────────────────

def to_int8(v):
    v = int(v) & 0xFF
    return v - 256 if v >= 128 else v

def wrap20(v):
    v = int(v) & 0xFFFFF
    return v - 0x100000 if v >= 0x80000 else v

def golden_2x2_relu(A, B):
    C = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            acc = sum(to_int8(A[i][k]) * to_int8(B[k][j]) for k in range(2))
            val = wrap20(acc)
            C[i][j] = 0 if val < 0 else val
    return C

def golden_4x4(A, B, relu=False, C_prev=None, accum=False):
    C = [[0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            acc = sum(to_int8(A[i][k]) * to_int8(B[k][j]) for k in range(4))
            val = wrap20(acc)
            if accum and C_prev is not None:
                val = wrap20(C_prev[i][j] + val)
            if relu and val < 0:
                val = 0
            C[i][j] = val
    return C

# ─── Helpers ──────────────────────────────────────────────────

async def do_reset(dut):
    dut.rst_n.value  = 0
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

def pack_2x2(A, B):
    flat = [A[0][0], A[0][1], A[1][0], A[1][1],
            B[0][0], B[0][1], B[1][0], B[1][1]]
    return [int(v) & 0xFF for v in flat]

def pack_4x4(A, B):
    flat = [A[r][c] for r in range(4) for c in range(4)] + \
           [B[r][c] for r in range(4) for c in range(4)]
    return [int(v) & 0xFF for v in flat]

def decode_output(raw, n_elements):
    out = []
    for i in range(n_elements):
        b0 = raw[i * 3 + 0] & 0xFF
        b1 = raw[i * 3 + 1] & 0xFF
        b2 = raw[i * 3 + 2] & 0x0F
        val = b0 | (b1 << 8) | (b2 << 16)
        if val & (1 << 19):
            val -= (1 << 20)
        out.append(val)
    return out

def flatten_2x2(C):
    return [C[r][c] for r in range(2) for c in range(2)]

def flatten_4x4(C):
    return [C[r][c] for r in range(4) for c in range(4)]

async def start_clock(dut):
    clock = Clock(dut.clk, 30, unit="ns")  # 33 MHz
    cocotb.start_soon(clock.start())

def assert_eq(got, exp, name):
    if got != exp:
        raise AssertionError(f"{name} FAIL:\n  got={got}\n  exp={exp}")

def is_done(dut):
    return bool(int(dut.uio_out.value) & 0x1)

# ─── Transaction runner ────────────────────────────────────────
#
# cocotb v2 disallows calling ReadOnly() when already in the ReadOnly
# phase (i.e. inside a loop that keeps awaiting ReadOnly).
# Solution: poll done ONLY on rising edges — signal values are stable
# immediately after the edge, so no ReadOnly needed.

async def run_transaction(dut, mode, input_bytes, n_out_elements, accum_clear=False):
    n_out_bytes = n_out_elements * 3

    # ── 1. START pulse ────────────────────────────────────────
    # Drive start=1 then clock edge → FSM IDLE → ST_LOAD
    dut.uio_in.value = 0x01 | (mode << 1) | ((1 << 3) if accum_clear else 0)
    dut.ui_in.value  = 0
    await RisingEdge(dut.clk)   # FSM transitions to ST_LOAD here

    # Deassert start immediately
    dut.uio_in.value = (mode << 1)

    # ── 2. LOAD phase ─────────────────────────────────────────
    # Drive each byte before its sampling edge
    for b in input_bytes:
        dut.ui_in.value = int(b) & 0xFF
        await RisingEdge(dut.clk)

    dut.ui_in.value = 0

    # ── 3. Wait for ST_OUTPUT (done goes high) ────────────────
    # Max wait: 8 compute cycles + a few spare
    for _ in range(16):
        await RisingEdge(dut.clk)
        if is_done(dut):
            break
    else:
        raise AssertionError("Timeout waiting for done to assert")

    # ── 4. Read output stream ─────────────────────────────────
    # done just went high after a rising edge; uo_out is already valid
    raw = [int(dut.uo_out.value) & 0xFF]

    for _ in range(n_out_bytes - 1):
        await RisingEdge(dut.clk)
        raw.append(int(dut.uo_out.value) & 0xFF)

    # ── 5. Wait for done to drop (back to IDLE) ───────────────
    for _ in range(n_out_bytes + 8):
        await RisingEdge(dut.clk)
        if not is_done(dut):
            break
    else:
        raise AssertionError("Timeout waiting for done to deassert")

    return decode_output(raw, n_out_elements)

# ═════════════════════════════════════════════════════════════
#  MODE 00: 2x2 + ReLU
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_00_all_zeros(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[0, 0], [0, 0]]; B = [[0, 0], [0, 0]]
    got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), "mode00 all_zeros")

@cocotb.test()
async def test_00_identity(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[127, 0], [0, 127]]; B = [[1, 0], [0, 1]]
    got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), "mode00 identity")

@cocotb.test()
async def test_00_relu_clamps_negatives(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[0, -1], [0, -1]]; B = [[1, 0], [1, 0]]
    got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), "mode00 relu_clamp")

@cocotb.test()
async def test_00_min_times_min(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[-128, 0], [0, -128]]; B = [[-128, 0], [0, -128]]
    got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), "mode00 min_times_min")

@cocotb.test()
async def test_00_one_hot_each_element(dut):
    await start_clock(dut); await do_reset(dut)
    B = [[3, 7], [11, 5]]
    for r, c in [(0,0),(0,1),(1,0),(1,1)]:
        A = [[0,0],[0,0]]; A[r][c] = 64
        got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
        assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), f"mode00 one_hot_A[{r}][{c}]")
    A = [[3, 7], [11, 5]]
    for r, c in [(0,0),(0,1),(1,0),(1,1)]:
        B2 = [[0,0],[0,0]]; B2[r][c] = 64
        got = await run_transaction(dut, 0, pack_2x2(A, B2), 4)
        assert_eq(got, flatten_2x2(golden_2x2_relu(A, B2)), f"mode00 one_hot_B[{r}][{c}]")

@cocotb.test()
async def test_00_mixed_signs(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[10, -20], [30, -40]]; B = [[-50, 60], [70, -80]]
    got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), "mode00 mixed_signs")

@cocotb.test()
async def test_00_random_regression(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(1001)
    for i in range(100):
        A = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        B = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
        assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), f"mode00 random[{i}]")

# ═════════════════════════════════════════════════════════════
#  MODE 01: 4x4 raw
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_01_all_zeros(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[0]*4 for _ in range(4)]; B = [[0]*4 for _ in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A, B)), "mode01 all_zeros")

@cocotb.test()
async def test_01_identity(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B = [[r*4+c+1 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A, B)), "mode01 identity")

@cocotb.test()
async def test_01_max_positive(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[127]*4 for _ in range(4)]; B = [[127]*4 for _ in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A, B)), "mode01 max_positive")

@cocotb.test()
async def test_01_min_times_min(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[-128]*4 for _ in range(4)]; B = [[-128]*4 for _ in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A, B)), "mode01 min_times_min")

@cocotb.test()
async def test_01_one_hot_each_element(dut):
    await start_clock(dut); await do_reset(dut)
    B = [[r*4+c+1 for c in range(4)] for r in range(4)]
    for r in range(4):
        for c in range(4):
            A = [[0]*4 for _ in range(4)]; A[r][c] = 1
            got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
            assert_eq(got, flatten_4x4(golden_4x4(A, B)), f"mode01 one_hot_A[{r}][{c}]")

@cocotb.test()
async def test_01_mixed_extremes(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(2002); choices = [-128,-1,0,1,127]
    for i in range(20):
        A = [[rng.choice(choices) for _ in range(4)] for _ in range(4)]
        B = [[rng.choice(choices) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
        assert_eq(got, flatten_4x4(golden_4x4(A, B)), f"mode01 extremes[{i}]")

@cocotb.test()
async def test_01_random_regression(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(3003)
    for i in range(50):
        A = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        B = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
        assert_eq(got, flatten_4x4(golden_4x4(A, B)), f"mode01 random[{i}]")

# ═════════════════════════════════════════════════════════════
#  MODE 10: 4x4 + ReLU
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_10_relu_clamps_negatives(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[-1 if r==c else 0 for c in range(4)] for r in range(4)]
    B = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 2, pack_4x4(A, B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A, B, relu=True)), "mode10 relu_clamps_neg")

@cocotb.test()
async def test_10_positive_unchanged(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B = [[r*4+c+1 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 2, pack_4x4(A, B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A, B, relu=True)), "mode10 pos_unchanged")

@cocotb.test()
async def test_10_random_regression(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(4004)
    for i in range(50):
        A = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        B = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 2, pack_4x4(A, B), 16)
        assert_eq(got, flatten_4x4(golden_4x4(A, B, relu=True)), f"mode10 random[{i}]")

# ═════════════════════════════════════════════════════════════
#  MODE 11: 4x4 tiled accumulate
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_11_clear_then_accumulate(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(5005)
    A1 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]
    B1 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]
    A2 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]
    B2 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]
    got1 = await run_transaction(dut, 3, pack_4x4(A1, B1), 16, accum_clear=True)
    C1_golden = golden_4x4(A1, B1)
    assert_eq(got1, flatten_4x4(C1_golden), "mode11 tile1")
    got2 = await run_transaction(dut, 3, pack_4x4(A2, B2), 16, accum_clear=False)
    assert_eq(got2, flatten_4x4(golden_4x4(A2, B2, C_prev=C1_golden, accum=True)), "mode11 tile2_accumulated")

@cocotb.test()
async def test_11_three_tile_chain(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(6006)
    tiles = [([[rng.randint(-50,50) for _ in range(4)] for _ in range(4)],
               [[rng.randint(-50,50) for _ in range(4)] for _ in range(4)])
             for _ in range(3)]
    C_running = None
    for idx, (A, B) in enumerate(tiles):
        got = await run_transaction(dut, 3, pack_4x4(A, B), 16, accum_clear=(idx==0))
        C_running = golden_4x4(A, B, C_prev=C_running, accum=(idx>0))
        assert_eq(got, flatten_4x4(C_running), f"mode11 tile{idx}")

@cocotb.test()
async def test_11_clear_resets_accumulator(dut):
    await start_clock(dut); await do_reset(dut)
    A = [[127]*4 for _ in range(4)]; B = [[127]*4 for _ in range(4)]
    await run_transaction(dut, 3, pack_4x4(A, B), 16, accum_clear=True)
    A2 = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B2 = [[r+1 if c==0 else 0 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 3, pack_4x4(A2, B2), 16, accum_clear=True)
    assert_eq(got, flatten_4x4(golden_4x4(A2, B2)), "mode11 clear_resets")

# ═════════════════════════════════════════════════════════════
#  PROTOCOL / FSM EDGE CASES
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_fsm_back_to_back_mode00(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(7007)
    for i in range(3):
        A = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        B = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
        assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), f"back2back_00 tx{i}")

@cocotb.test()
async def test_fsm_back_to_back_mode01(dut):
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(8008)
    for i in range(2):
        A = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        B = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 1, pack_4x4(A, B), 16)
        assert_eq(got, flatten_4x4(golden_4x4(A, B)), f"back2back_01 tx{i}")

@cocotb.test()
async def test_fsm_mode_switch(dut):
    await start_clock(dut); await do_reset(dut)
    A2 = [[10,-5],[3,7]]; B2 = [[2,0],[1,-1]]
    got2 = await run_transaction(dut, 0, pack_2x2(A2, B2), 4)
    assert_eq(got2, flatten_2x2(golden_2x2_relu(A2, B2)), "mode_switch 2x2")
    A4 = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B4 = [[r*4+c+1 for c in range(4)] for r in range(4)]
    got4 = await run_transaction(dut, 1, pack_4x4(A4, B4), 16)
    assert_eq(got4, flatten_4x4(golden_4x4(A4, B4)), "mode_switch 4x4")

@cocotb.test()
async def test_fsm_reset_recovery(dut):
    await start_clock(dut); await do_reset(dut)
    dut.uio_in.value = 0x01
    dut.ui_in.value  = 0
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0x00
    dut.ui_in.value  = 0x55
    await RisingEdge(dut.clk)
    dut.ui_in.value  = 0xAA
    await RisingEdge(dut.clk)
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value  = 1
    dut.uio_in.value = 0
    dut.ui_in.value  = 0
    await ClockCycles(dut.clk, 2)
    A = [[3,7],[2,-5]]; B = [[4,-1],[6,2]]
    got = await run_transaction(dut, 0, pack_2x2(A, B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A, B)), "reset_recovery")