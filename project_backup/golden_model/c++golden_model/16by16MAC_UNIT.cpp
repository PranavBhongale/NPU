/*
 * ============================================================
 *   16x16 MAC ARRAY — Black-Box Functional Simulation
 * ============================================================
 *
 * Interface contract (AXI-Stream style):
 *
 *   Producer drives:  valid_in, A[][], B[][], bias[]
 *   Consumer drives:  ready_out  (module signals it can accept)
 *   Module drives:    valid_out, C[][]
 *   Upstream drives:  ready_in   (downstream signals it can accept)
 *
 * Handshake rule:
 *   A transfer occurs ONLY when (valid & ready) == true.
 * Pipeline FSM:
 *
 *   IDLE ──(valid_in & ready_out)──► COMPUTE ──(done)──► OUTPUT
 *     ▲                                                      │
 *     └──────────────(ready_in asserted)────────────────────-┘
 *
 * ============================================================
 */

#include <iostream>
#include <cstdint>
#include <cstring>
#include <iomanip>

using namespace std;

/* ============================================================
 *  Processing Element
 * ============================================================
 *  Performs:  accumulator += A * B
 *  One PE per output element C[row][col]
 * ============================================================ */

class PE {

public:

    int32_t accumulator = 0;
    bool    output_valid = false;

    void clear() {
        accumulator  = 0;
        output_valid = false;
    }

    // Single MAC step: accumulate A*B if upstream is valid
    void mac(int8_t a, int8_t b, bool upstream_valid) {

        if (upstream_valid) {
            accumulator  += static_cast<int32_t>(a) * static_cast<int32_t>(b);
            output_valid  = true;
        }
    }

    void add_bias(int32_t bias) {
        accumulator += bias;
    }

    int32_t read() const {
        return accumulator;
    }
};

/* ============================================================
 *  MAC_ARRAY — Black-Box Interface
 * ============================================================
 *
 *  Ports
 *  ─────
 *  Input:
 *    A[SIZE][SIZE]   — INT8 activation matrix
 *    B[SIZE][SIZE]   — INT8 weight matrix
 *    bias[SIZE]      — INT32 bias vector (per output column)
 *    valid_in        — producer has valid A/B/bias
 *    ready_in        — downstream can accept C
 *
 *  Output:
 *    C[SIZE][SIZE]   — INT32 result  (C = A×B + bias)
 *    valid_out       — C is valid and ready for downstream
 *    ready_out       — module can accept new A/B inputs
 *                      (LOW while computing or holding output)
 *
 * ============================================================ */

class MAC_ARRAY {

public:

    static const int SIZE = 16;

    // ── Input ports ──────────────────────────────────────────
    int8_t   A[SIZE][SIZE]   = {};
    int8_t   B[SIZE][SIZE]   = {};
    int32_t  bias[SIZE]      = {};

    bool     valid_in  = false;  // producer: data on A/B/bias is valid
    bool     ready_in  = false;  // downstream: ready to accept C

    // ── Output ports ─────────────────────────────────────────
    int32_t  C[SIZE][SIZE]   = {};

    bool     valid_out = false;  // C is valid
    bool     ready_out = true;   // module is ready for new input

    // ── Internal state ────────────────────────────────────────
    enum class State { IDLE, COMPUTE, OUTPUT };

private:

    State    state_     = State::IDLE;
    PE       pe_[SIZE][SIZE];

    // Latched copies of A, B, bias (captured on handshake)
    int8_t   A_lat_[SIZE][SIZE] = {};
    int8_t   B_lat_[SIZE][SIZE] = {};
    int32_t  bias_lat_[SIZE]    = {};

    int      step_ = 0;  // current k-index in the MAC loop

public:

    MAC_ARRAY() { reset(); }

    /* ----------------------------------------------------------
     *  reset() — hard reset: clear everything, go to IDLE
     * ---------------------------------------------------------- */
    void reset() {

        state_     = State::IDLE;
        valid_out  = false;
        ready_out  = true;
        step_      = 0;

        memset(C,       0, sizeof(C));
        memset(A_lat_,  0, sizeof(A_lat_));
        memset(B_lat_,  0, sizeof(B_lat_));
        memset(bias_lat_,  0, sizeof(bias_lat_));

        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                pe_[i][j].clear();
    }

    /* ----------------------------------------------------------
     *  tick() — advance the FSM by one clock cycle
     *
     *  Call this once per simulated cycle after setting
     *  the input port values (valid_in, A, B, bias, ready_in).
     * ---------------------------------------------------------- */
    void tick() {

        switch (state_) {

        // ──────────────────────────────────────────────────────
        case State::IDLE:
        // ──────────────────────────────────────────────────────

            ready_out = true;
            valid_out = false;

            // Handshake fires: latch inputs and move to COMPUTE
            if (valid_in && ready_out) {

                _latch_inputs();

                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        pe_[i][j].clear();

                step_     = 0;
                ready_out = false;   // block new input while computing
                state_    = State::COMPUTE;

                cout << "[MAC_ARRAY] IDLE → COMPUTE  (inputs latched)\n";
            }
            break;

        // ──────────────────────────────────────────────────────
        case State::COMPUTE:
        // ──────────────────────────────────────────────────────

            // One MAC step per cycle (k-index = step_)
            for (int row = 0; row < SIZE; row++)
                for (int col = 0; col < SIZE; col++)
                    pe_[row][col].mac(A_lat_[row][step_],
                                      B_lat_[step_][col],
                                      /*upstream_valid=*/true);

            step_++;

            if (step_ == SIZE) {

                // Add bias and commit to output port
                for (int row = 0; row < SIZE; row++)
                    for (int col = 0; col < SIZE; col++) {
                        pe_[row][col].add_bias(bias_lat_[col]);
                        C[row][col] = pe_[row][col].read();
                    }

                valid_out = true;
                state_    = State::OUTPUT;

                cout << "[MAC_ARRAY] COMPUTE → OUTPUT  (result valid)\n";
            }
            break;

        // ──────────────────────────────────────────────────────
        case State::OUTPUT:
        // ──────────────────────────────────────────────────────

            // Hold C valid until downstream asserts ready_in
            if (ready_in && valid_out) {

                valid_out = false;
                ready_out = true;   // accept next transaction
                state_    = State::IDLE;

                cout << "[MAC_ARRAY] OUTPUT → IDLE   (C consumed)\n";
            }
            break;
        }
    }

    /* ----------------------------------------------------------
     *  status() — print current FSM / port state
     * ---------------------------------------------------------- */
    void status() const {

        const char* snames[] = { "IDLE", "COMPUTE", "OUTPUT" };
        cout << "  state="     << snames[static_cast<int>(state_)]
             << "  step="      << step_
             << "  valid_in="  << valid_in
             << "  ready_out=" << ready_out
             << "  valid_out=" << valid_out
             << "  ready_in="  << ready_in
             << "\n";
    }

    /* ----------------------------------------------------------
     *  print_output() — formatted matrix dump
     * ---------------------------------------------------------- */
    void print_output() const {

        cout << "\n===== C = A x B + bias =====\n";
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++)
                cout << setw(6) << C[i][j];
            cout << "\n";
        }
    }

private:

    void _latch_inputs() {
        memcpy(A_lat_,    A,    sizeof(A));
        memcpy(B_lat_,    B,    sizeof(B));
        memcpy(bias_lat_, bias, sizeof(bias));
    }
};

/* ============================================================
 *  Testbench
 *
 *  Drives the black-box interface cycle by cycle.
 *  Tests:
 *    1. Basic handshake  — A=1, B=2, bias=1  → C[i][j] = 33
 *    2. Back-pressure    — downstream deasserts ready_in for
 *                          2 cycles; result must be held
 *    3. Pipeline re-use  — second transaction immediately
 *                          after first is consumed
 * ============================================================ */

int main() {

    MAC_ARRAY mac;
    int cycle = 0;

    auto tick_cycle = [&]() {
        cout << "Cycle " << setw(3) << cycle++ << ": ";
        mac.status();
        mac.tick();
    };

    /* ── Test 1: basic transaction ─────────────────────────── */
    cout << "\n=== TEST 1: Basic handshake (A=1, B=2, bias=1) ===\n";

    for (int i = 0; i < MAC_ARRAY::SIZE; i++) {
        mac.bias[i] = 1;
        for (int j = 0; j < MAC_ARRAY::SIZE; j++) {
            mac.A[i][j] = 1;
            mac.B[i][j] = 2;
        }
    }

    mac.valid_in = true;   // producer: data ready
    mac.ready_in = true;   // downstream: ready to accept result

    // Cycle 0 — IDLE: handshake fires, transition to COMPUTE
    tick_cycle();

    // Lower valid_in (data was latched; no new data yet)
    mac.valid_in = false;

    // COMPUTE phase: SIZE cycles (one k-step each)
    for (int k = 0; k < MAC_ARRAY::SIZE; k++)
        tick_cycle();

    // OUTPUT phase: one cycle to consume
    tick_cycle();

    // One extra idle cycle
    tick_cycle();

    // Verify: each cell should be  1*2*16 + 1 = 33
    bool pass = true;
    for (int i = 0; i < MAC_ARRAY::SIZE && pass; i++)
        for (int j = 0; j < MAC_ARRAY::SIZE && pass; j++)
            if (mac.C[i][j] != 33) { pass = false; }

    cout << "Test 1: " << (pass ? "PASS ✓" : "FAIL ✗")
         << "  (expected 33, got " << mac.C[0][0] << ")\n";
    mac.print_output();

    /* ── Test 2: back-pressure ─────────────────────────────── */
    cout << "\n=== TEST 2: Back-pressure (downstream not ready) ===\n";
    mac.reset(); cycle = 0;

    for (int i = 0; i < MAC_ARRAY::SIZE; i++) {
        mac.bias[i] = 0;
        for (int j = 0; j < MAC_ARRAY::SIZE; j++) {
            mac.A[i][j] = 2;
            mac.B[i][j] = 3;
        }
    }

    mac.valid_in = true;
    mac.ready_in = false;  // downstream NOT ready yet

    tick_cycle();              // handshake fires, COMPUTE begins
    mac.valid_in = false;

    for (int k = 0; k < MAC_ARRAY::SIZE; k++)
        tick_cycle();

    // Module is now in OUTPUT, holding valid_out=true
    // Back-pressure: keep ready_in=false for 2 cycles
    for (int bp = 0; bp < 2; bp++) {
        cout << "  [back-pressure cycle " << bp << "] ";
        tick_cycle();
    }

    // Now downstream is ready
    mac.ready_in = true;
    tick_cycle();  // OUTPUT → IDLE
    tick_cycle();  // idle

    bool pass2 = true;
    for (int i = 0; i < MAC_ARRAY::SIZE && pass2; i++)
        for (int j = 0; j < MAC_ARRAY::SIZE && pass2; j++)
            if (mac.C[i][j] != 96) pass2 = false;

    cout << "Test 2: " << (pass2 ? "PASS ✓" : "FAIL ✗")
         << "  (expected 96, got " << mac.C[0][0] << ")\n";

    /* ── Test 3: pipeline re-use ───────────────────────────── */
    cout << "\n=== TEST 3: Second transaction immediately after first ===\n";
    mac.reset(); cycle = 0;

    // First transaction: A=1, B=1, bias=0 → C=16
    for (int i = 0; i < MAC_ARRAY::SIZE; i++) {
        mac.bias[i] = 0;
        for (int j = 0; j < MAC_ARRAY::SIZE; j++) {
            mac.A[i][j] = 1;
            mac.B[i][j] = 1;
        }
    }
    mac.valid_in = true;
    mac.ready_in = true;
    tick_cycle(); mac.valid_in = false;
    for (int k = 0; k < MAC_ARRAY::SIZE; k++) tick_cycle();
    tick_cycle(); // OUTPUT → IDLE

    // Second transaction immediately: A=3, B=3, bias=5 → C=149
    for (int i = 0; i < MAC_ARRAY::SIZE; i++) {
        mac.bias[i] = 5;
        for (int j = 0; j < MAC_ARRAY::SIZE; j++) {
            mac.A[i][j] = 3;
            mac.B[i][j] = 3;
        }
    }
    mac.valid_in = true;
    tick_cycle(); mac.valid_in = false;
    for (int k = 0; k < MAC_ARRAY::SIZE; k++) tick_cycle();
    tick_cycle(); // OUTPUT → IDLE
    tick_cycle();

    bool pass3 = true;
    for (int i = 0; i < MAC_ARRAY::SIZE && pass3; i++)
        for (int j = 0; j < MAC_ARRAY::SIZE && pass3; j++)
            if (mac.C[i][j] != 149) pass3 = false;

    cout << "Test 3: " << (pass3 ? "PASS ✓" : "FAIL ✗")
         << "  (expected 149, got " << mac.C[0][0] << ")\n";

    cout << "\n=== ALL TESTS DONE ===\n";
    return (pass && pass2 && pass3) ? 0 : 1;
}



