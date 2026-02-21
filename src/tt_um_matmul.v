// ============================================================
//  tt_um_matmul.v  –  4-Mode Matrix Multiply Accelerator
//  TinyTapeout SKY130  (max 16 tiles @ 33 MHz)
//
//  Modes  uio_in[2:1]:
//    00  2x2 matmul + ReLU      1 compute cycle,  12 out bytes
//    01  4x4 matmul raw         8 compute cycles, 48 out bytes
//    10  4x4 matmul + ReLU      8 compute cycles, 48 out bytes
//    11  4x4 tiled accumulate   8 compute cycles, 48 out bytes
//                               C += A*B each call
//
//  Number format:
//    Inputs  : signed int8  (-128..127)
//    Outputs : signed int20 (natural 20-bit two's-complement wrap)
//    Packing : byte0=result[7:0], byte1=result[15:8],
//              byte2={4'b0, result[19:16]}
//
//  Hardware: 8 shared signed int8 multipliers, reused per pass
//
//  Pin protocol:
//    ui_in[7:0]   data byte in  (LOAD phase)
//    uo_out[7:0]  data byte out (OUTPUT phase), else 0
//    uio_in[0]    start         (1-cycle pulse)
//    uio_in[2:1]  mode[1:0]    (sampled at start)
//    uio_in[3]    accum_clear  (mode 11: 1=zero C before tile)
//    uio_out[0]   done         (high throughout OUTPUT phase)
//    uio_oe[0]    1            (only bit 0 is an output)
//
//  LOAD byte order:
//    2x2  8 bytes: A[0][0] A[0][1] A[1][0] A[1][1]
//                  B[0][0] B[0][1] B[1][0] B[1][1]
//    4x4 32 bytes: A row-major A[0][0..3] A[1][0..3] A[2][0..3] A[3][0..3]
//                  B row-major B[0][0..3] B[1][0..3] B[2][0..3] B[3][0..3]
//
//  OUTPUT byte order: C row-major, 3 bytes per element (LSB first)
// ============================================================
`default_nettype none

module tt_um_matmul (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    // Suppress unused-port warnings (TinyTapeout requirement)
    wire _unused = &{ena, uio_in[7:4], 1'b0};

    // uio[0] is output (done flag); rest are inputs
    assign uio_oe  = 8'b0000_0001;
    assign uio_out = {7'b0, done_r};

    // ── FSM states ────────────────────────────────────────────
    localparam [1:0] ST_IDLE    = 2'd0,
                     ST_LOAD    = 2'd1,
                     ST_COMPUTE = 2'd2,
                     ST_OUTPUT  = 2'd3;

    // ── Mode constants ────────────────────────────────────────
    localparam [1:0] M_2x2R = 2'b00,   // 2x2 + ReLU
                     M_4x4  = 2'b01,   // 4x4 raw
                     M_4x4R = 2'b10,   // 4x4 + ReLU
                     M_4x4A = 2'b11;   // 4x4 accumulate

    // ── Input decoding ────────────────────────────────────────
    wire       start = uio_in[0];
    wire [1:0] mode_in = uio_in[2:1];
    wire       aclr    = uio_in[3];  // accumulator clear for mode 11

    // ── State registers ───────────────────────────────────────
    reg [1:0] state, mode_r;
    reg       done_r;
    reg [4:0] ld_cnt;   // load byte counter  0..31
    reg [2:0] pass_r;   // compute pass index 0..7
    reg [3:0] oel;      // output element     0..15
    reg [1:0] osub;     // output sub-byte    0..2

    // ── Matrix storage (flat row-major) ───────────────────────
    // A[r][c] = matA[r*N+c],  B[r][c] = matB[r*N+c]
    // 2x2: N=2, indices 0..3
    // 4x4: N=4, indices 0..15
    reg signed [7:0]  matA [0:15];
    reg signed [7:0]  matB [0:15];
    reg signed [19:0] matC [0:15];  // 20-bit result registers

    // ── Derived control signals ───────────────────────────────
    wire is2   = (mode_r == M_2x2R);
    wire isAcc = (mode_r == M_4x4A);
    wire rlEn  = (mode_r == M_2x2R) || (mode_r == M_4x4R);

    // ── Multiplier input mux (fully combinational) ────────────
    //
    // 8 multipliers: p[k] = mA[k] * mB[k]
    //
    // 2x2 mode (all 4 outputs computed simultaneously):
    //   mA[0,1] x mB[0,1]  → s00 = C[0][0]  (A row0 · B col0)
    //   mA[2,3] x mB[2,3]  → s01 = C[0][1]  (A row0 · B col1)
    //   mA[4,5] x mB[4,5]  → s10 = C[1][0]  (A row1 · B col0)
    //   mA[6,7] x mB[6,7]  → s11 = C[1][1]  (A row1 · B col1)
    //
    // 4x4 mode (2 outputs per pass):
    //   Each pass p: row r=p>>1, col_lo=2*(p&1), col_hi=col_lo+1
    //   mA[0..3] x mB[0..3] → slo  (4 products summed → C[r][col_lo])
    //   mA[4..7] x mB[4..7] → shi  (4 products summed → C[r][col_hi])
    //
    // B is stored row-major: B[r][c] = matB[r*4+c]
    // So B column c products: matB[0*4+c], matB[1*4+c], matB[2*4+c], matB[3*4+c]

    reg signed [7:0] mA [0:7];
    reg signed [7:0] mB [0:7];

    always @(*) begin : MUX
        integer k;
        for (k = 0; k < 8; k = k + 1) begin
            mA[k] = 8'sd0;
            mB[k] = 8'sd0;
        end

        if (is2) begin
            // ── 2x2 fixed wiring ──────────────────────────────
            // C[0][0] = A[0]*B[0] + A[1]*B[2]  (row0·col0)
            mA[0]=matA[0]; mB[0]=matB[0];
            mA[1]=matA[1]; mB[1]=matB[2];
            // C[0][1] = A[0]*B[1] + A[1]*B[3]  (row0·col1)
            mA[2]=matA[0]; mB[2]=matB[1];
            mA[3]=matA[1]; mB[3]=matB[3];
            // C[1][0] = A[2]*B[0] + A[3]*B[2]  (row1·col0)
            mA[4]=matA[2]; mB[4]=matB[0];
            mA[5]=matA[3]; mB[5]=matB[2];
            // C[1][1] = A[2]*B[1] + A[3]*B[3]  (row1·col1)
            mA[6]=matA[2]; mB[6]=matB[1];
            mA[7]=matA[3]; mB[7]=matB[3];
        end else begin
            // ── 4x4 pass-based wiring ─────────────────────────
            // pass 0: r=0 cols 0,1   pass 1: r=0 cols 2,3
            // pass 2: r=1 cols 0,1   pass 3: r=1 cols 2,3
            // pass 4: r=2 cols 0,1   pass 5: r=2 cols 2,3
            // pass 6: r=3 cols 0,1   pass 7: r=3 cols 2,3
            case (pass_r)
                3'd0: begin  // row 0, cols 0 & 1
                    mA[0]=matA[0];  mB[0]=matB[0];
                    mA[1]=matA[1];  mB[1]=matB[4];
                    mA[2]=matA[2];  mB[2]=matB[8];
                    mA[3]=matA[3];  mB[3]=matB[12];
                    mA[4]=matA[0];  mB[4]=matB[1];
                    mA[5]=matA[1];  mB[5]=matB[5];
                    mA[6]=matA[2];  mB[6]=matB[9];
                    mA[7]=matA[3];  mB[7]=matB[13];
                end
                3'd1: begin  // row 0, cols 2 & 3
                    mA[0]=matA[0];  mB[0]=matB[2];
                    mA[1]=matA[1];  mB[1]=matB[6];
                    mA[2]=matA[2];  mB[2]=matB[10];
                    mA[3]=matA[3];  mB[3]=matB[14];
                    mA[4]=matA[0];  mB[4]=matB[3];
                    mA[5]=matA[1];  mB[5]=matB[7];
                    mA[6]=matA[2];  mB[6]=matB[11];
                    mA[7]=matA[3];  mB[7]=matB[15];
                end
                3'd2: begin  // row 1, cols 0 & 1
                    mA[0]=matA[4];  mB[0]=matB[0];
                    mA[1]=matA[5];  mB[1]=matB[4];
                    mA[2]=matA[6];  mB[2]=matB[8];
                    mA[3]=matA[7];  mB[3]=matB[12];
                    mA[4]=matA[4];  mB[4]=matB[1];
                    mA[5]=matA[5];  mB[5]=matB[5];
                    mA[6]=matA[6];  mB[6]=matB[9];
                    mA[7]=matA[7];  mB[7]=matB[13];
                end
                3'd3: begin  // row 1, cols 2 & 3
                    mA[0]=matA[4];  mB[0]=matB[2];
                    mA[1]=matA[5];  mB[1]=matB[6];
                    mA[2]=matA[6];  mB[2]=matB[10];
                    mA[3]=matA[7];  mB[3]=matB[14];
                    mA[4]=matA[4];  mB[4]=matB[3];
                    mA[5]=matA[5];  mB[5]=matB[7];
                    mA[6]=matA[6];  mB[6]=matB[11];
                    mA[7]=matA[7];  mB[7]=matB[15];
                end
                3'd4: begin  // row 2, cols 0 & 1
                    mA[0]=matA[8];  mB[0]=matB[0];
                    mA[1]=matA[9];  mB[1]=matB[4];
                    mA[2]=matA[10]; mB[2]=matB[8];
                    mA[3]=matA[11]; mB[3]=matB[12];
                    mA[4]=matA[8];  mB[4]=matB[1];
                    mA[5]=matA[9];  mB[5]=matB[5];
                    mA[6]=matA[10]; mB[6]=matB[9];
                    mA[7]=matA[11]; mB[7]=matB[13];
                end
                3'd5: begin  // row 2, cols 2 & 3
                    mA[0]=matA[8];  mB[0]=matB[2];
                    mA[1]=matA[9];  mB[1]=matB[6];
                    mA[2]=matA[10]; mB[2]=matB[10];
                    mA[3]=matA[11]; mB[3]=matB[14];
                    mA[4]=matA[8];  mB[4]=matB[3];
                    mA[5]=matA[9];  mB[5]=matB[7];
                    mA[6]=matA[10]; mB[6]=matB[11];
                    mA[7]=matA[11]; mB[7]=matB[15];
                end
                3'd6: begin  // row 3, cols 0 & 1
                    mA[0]=matA[12]; mB[0]=matB[0];
                    mA[1]=matA[13]; mB[1]=matB[4];
                    mA[2]=matA[14]; mB[2]=matB[8];
                    mA[3]=matA[15]; mB[3]=matB[12];
                    mA[4]=matA[12]; mB[4]=matB[1];
                    mA[5]=matA[13]; mB[5]=matB[5];
                    mA[6]=matA[14]; mB[6]=matB[9];
                    mA[7]=matA[15]; mB[7]=matB[13];
                end
                default: begin  // 3'd7: row 3, cols 2 & 3
                    mA[0]=matA[12]; mB[0]=matB[2];
                    mA[1]=matA[13]; mB[1]=matB[6];
                    mA[2]=matA[14]; mB[2]=matB[10];
                    mA[3]=matA[15]; mB[3]=matB[14];
                    mA[4]=matA[12]; mB[4]=matB[3];
                    mA[5]=matA[13]; mB[5]=matB[7];
                    mA[6]=matA[14]; mB[6]=matB[11];
                    mA[7]=matA[15]; mB[7]=matB[15];
                end
            endcase
        end
    end  // always MUX

    // ── 8 signed int8 multipliers ─────────────────────────────
    wire signed [15:0] p0 = mA[0] * mB[0];
    wire signed [15:0] p1 = mA[1] * mB[1];
    wire signed [15:0] p2 = mA[2] * mB[2];
    wire signed [15:0] p3 = mA[3] * mB[3];
    wire signed [15:0] p4 = mA[4] * mB[4];
    wire signed [15:0] p5 = mA[5] * mB[5];
    wire signed [15:0] p6 = mA[6] * mB[6];
    wire signed [15:0] p7 = mA[7] * mB[7];

    // Sign-extend products to 20 bits before accumulation
    wire signed [19:0] e0={{4{p0[15]}},p0}, e1={{4{p1[15]}},p1},
                       e2={{4{p2[15]}},p2}, e3={{4{p3[15]}},p3},
                       e4={{4{p4[15]}},p4}, e5={{4{p5[15]}},p5},
                       e6={{4{p6[15]}},p6}, e7={{4{p7[15]}},p7};

    // ── Partial sums ──────────────────────────────────────────
    // 2x2: 4 pairs
    wire signed [19:0] s00 = e0+e1;
    wire signed [19:0] s01 = e2+e3;
    wire signed [19:0] s10 = e4+e5;
    wire signed [19:0] s11 = e6+e7;
    // 4x4: 2 quads
    wire signed [19:0] slo = e0+e1+e2+e3;
    wire signed [19:0] shi = e4+e5+e6+e7;

    // ── ReLU output wires ─────────────────────────────────────
    // Clamp negative values to 0 when ReLU is enabled
    wire signed [19:0] Cout [0:15];
    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin : RL
            assign Cout[g] = (rlEn && matC[g][19]) ? 20'sd0 : matC[g];
        end
    endgenerate

    // ── Output byte mux ───────────────────────────────────────
    reg signed [19:0] oval;
    always @(*) begin : OMUX
        case (oel)
            4'd0:  oval = Cout[0];   4'd1:  oval = Cout[1];
            4'd2:  oval = Cout[2];   4'd3:  oval = Cout[3];
            4'd4:  oval = Cout[4];   4'd5:  oval = Cout[5];
            4'd6:  oval = Cout[6];   4'd7:  oval = Cout[7];
            4'd8:  oval = Cout[8];   4'd9:  oval = Cout[9];
            4'd10: oval = Cout[10];  4'd11: oval = Cout[11];
            4'd12: oval = Cout[12];  4'd13: oval = Cout[13];
            4'd14: oval = Cout[14];  default: oval = Cout[15];
        endcase
    end

    reg [7:0] obyte;
    always @(*) begin : BMUX
        case (osub)
            2'd0:    obyte = oval[7:0];
            2'd1:    obyte = oval[15:8];
            default: obyte = {4'b0000, oval[19:16]};
        endcase
    end

    assign uo_out = (state == ST_OUTPUT) ? obyte : 8'b0;

    // ── FSM ───────────────────────────────────────────────────
    wire [3:0] max_oel = is2 ? 4'd3 : 4'd15;

    integer fi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= ST_IDLE; mode_r <= 2'b0; done_r <= 1'b0;
            ld_cnt <= 5'd0; pass_r <= 3'd0; oel <= 4'd0; osub <= 2'd0;
            for (fi = 0; fi < 16; fi = fi + 1) begin
                matA[fi] <= 8'sd0;
                matB[fi] <= 8'sd0;
                matC[fi] <= 20'sd0;
            end
        end else begin
            case (state)

                // ── IDLE ──────────────────────────────────────
                ST_IDLE: begin
                    done_r <= 1'b0;
                    if (start) begin
                        mode_r <= mode_in;
                        ld_cnt <= 5'd0;
                        pass_r <= 3'd0;
                        oel    <= 4'd0;
                        osub   <= 2'd0;
                        // Mode 11: clear accumulator if requested
                        if (mode_in == M_4x4A && aclr) begin
                            for (fi = 0; fi < 16; fi = fi + 1)
                                matC[fi] <= 20'sd0;
                        end
                        state <= ST_LOAD;
                    end
                end

                // ── LOAD ──────────────────────────────────────
                // 2x2: 8 bytes  → matA[0..3] then matB[0..3]
                // 4x4: 32 bytes → matA[0..15] then matB[0..15]
                ST_LOAD: begin
                    if (is2) begin
                        if (ld_cnt < 5'd4)
                            matA[ld_cnt[1:0]] <= $signed(ui_in);
                        else
                            matB[ld_cnt[1:0]] <= $signed(ui_in);
                        if (ld_cnt == 5'd7) begin
                            ld_cnt <= 5'd0;
                            state  <= ST_COMPUTE;
                        end else
                            ld_cnt <= ld_cnt + 5'd1;
                    end else begin
                        if (ld_cnt < 5'd16)
                            matA[ld_cnt[3:0]] <= $signed(ui_in);
                        else
                            matB[ld_cnt[3:0]] <= $signed(ui_in);
                        if (ld_cnt == 5'd31) begin
                            ld_cnt <= 5'd0;
                            state  <= ST_COMPUTE;
                        end else
                            ld_cnt <= ld_cnt + 5'd1;
                    end
                end

                // ── COMPUTE ───────────────────────────────────
                ST_COMPUTE: begin
                    if (is2) begin
                        // 2x2: all 4 outputs in 1 cycle
                        matC[0] <= s00;
                        matC[1] <= s01;
                        matC[2] <= s10;
                        matC[3] <= s11;
                        done_r  <= 1'b1;
                        state   <= ST_OUTPUT;
                    end else begin
                        // 4x4: 8 passes — 2 outputs per cycle
                        // {pass_r, 1'b0} = pass_r*2 (lo index)
                        // {pass_r, 1'b1} = pass_r*2+1 (hi index)
                        if (isAcc) begin
                            matC[{pass_r,1'b0}] <= matC[{pass_r,1'b0}] + slo;
                            matC[{pass_r,1'b1}] <= matC[{pass_r,1'b1}] + shi;
                        end else begin
                            matC[{pass_r,1'b0}] <= slo;
                            matC[{pass_r,1'b1}] <= shi;
                        end
                        if (pass_r == 3'd7) begin
                            pass_r <= 3'd0;
                            done_r <= 1'b1;
                            state  <= ST_OUTPUT;
                        end else
                            pass_r <= pass_r + 3'd1;
                    end
                end

                // ── OUTPUT ────────────────────────────────────
                // Streams 3 bytes per element, elements 0..max_oel
                ST_OUTPUT: begin
                    done_r <= 1'b1;
                    if (osub == 2'd2) begin
                        osub <= 2'd0;
                        if (oel == max_oel) begin
                            oel    <= 4'd0;
                            done_r <= 1'b0;
                            state  <= ST_IDLE;
                        end else
                            oel <= oel + 4'd1;
                    end else
                        osub <= osub + 2'd1;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule

