import os
import struct
import unittest

# ─────────────── Simulation constants ───────────────
W            = 15             # grid width
H            = 15             # grid height
MAXP_CELL    = 3             # max people per cell
NP           = int(W * H * MAXP_CELL * 0.5)  # total people
INFP         = 0.5           # initial infected fraction
IMM          = 0.1           # initial immune fraction
ND           = 20            # number of days
INCUBATION_DAYS   = 4             # incubation days
SIM_EXE      = "./bin/plague"       # path to your simulator executable
REPORT_DIR   = "report"      # where day_*.dat files live
# ─────────────────────────────────────────────────────

class SimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.days = []
        for day in range(ND):
            fn = os.path.join(REPORT_DIR, f"day_{day:03d}.dat")
            if not os.path.isfile(fn):
                raise FileNotFoundError(f"Expected report file not found: {fn}")
            with open(fn, "rb") as f:
                np_val = struct.unpack("i", f.read(4))[0]
                if np_val != NP:
                    raise ValueError(f"Day {day:03d}: header NP={np_val} != day0 NP={NP}")
                people = [struct.unpack("iii", f.read(12)) for _ in range(NP)]
            cls.days.append(people)


    def test_total_entries_per_day(self):
        for d, people in enumerate(self.days):
            self.assertEqual(len(people), NP,
                f"Day {d}: expected {NP} entries, got {len(people)}")

    def test_state_counts_sum_to_NP(self):
        for d, people in enumerate(self.days):
            counts = {0:0,1:0,2:0,3:0}
            for _, _, state in people:
                counts[state] += 1
            total = sum(counts.values())
            self.assertEqual(total, NP,
                f"Day {d}: sum of state counts {total} != NP ({NP})")

    def test_cell_occupancy_limit(self):
        for d, people in enumerate(self.days):
            occ = {}
            for x, y, _ in people:
                if x >= 0 and y >= 0:
                    occ[(x,y)] = occ.get((x,y), 0) + 1
            for (x,y), c in occ.items():
                self.assertLessEqual(c, MAXP_CELL,
                    f"Day {d}: cell ({x},{y}) has {c} > MAXP_CELL ({MAXP_CELL})")

    def test_positions_valid_or_dead(self):
        for d, people in enumerate(self.days):
            for i, (x, y, _) in enumerate(people):
                valid = (x == -1 and y == -1) or (0 <= x < W and 0 <= y < H)
                self.assertTrue(valid,
                    f"Day {d}, person {i}: invalid position ({x},{y})")

    def test_max_one_cell_movement(self):
        for d in range(1, len(self.days)):
            prev = self.days[d-1]
            curr = self.days[d]
            for i, (x0, y0, _) in enumerate(prev):
                x1, y1, _ = curr[i]
                if x0 >= 0 and y0 >= 0 and x1 >= 0 and y1 >= 0:
                    self.assertLessEqual(abs(x1 - x0), 1,
                        f"Day {d}, person {i}: moved Δx = {x1-x0} > 1")
                    self.assertLessEqual(abs(y1 - y0), 1,
                        f"Day {d}, person {i}: moved Δy = {y1-y0} > 1")

    def test_deaths_never_decrease(self):
        prev_dead = 0
        for d, people in enumerate(self.days):
            dead = sum(1 for x,y,s in people if s == 3)
            self.assertGreaterEqual(dead, prev_dead,
                f"Day {d}: dead {dead} < previous day dead {prev_dead}")
            prev_dead = dead

    def test_initial_immune_and_infected_counts(self):
        day0 = self.days[0]
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for x, y, state in day0:
            counts[state] += 1

        expected_immune   = int(NP * IMM)
        expected_infected = int(NP * INFP)

        self.assertEqual(counts[0], expected_immune,
                         f"Day 0: expected {expected_immune} immune, got {counts[0]}")
        self.assertEqual(counts[1], expected_infected,
                         f"Day 0: expected {expected_infected} infected, got {counts[1]}")
    
    def test_incubation_duration_and_transition(self):
        for i in range(NP):
            history = [day[i][2] for day in self.days]

            run_length = 0
            for day_idx, state in enumerate(history):
                if state == 1:  # INFECTED
                    run_length += 1
                    self.assertLessEqual(
                        run_length, INCUBATION_DAYS+5,
                        f"Person {i}: infected for {run_length} days by day {day_idx}, exceeds INCUBATION_DAYS={INCUBATION_DAYS}"
                    )
                else:
                    if run_length > 0:
                        # immediately after an infected run, state must be immune (0) or dead (3) or recovered (2)
                        self.assertIn(
                            state, (0, 2, 3),
                            f"Person {i}: after {run_length} infected days, day {day_idx} state is {state}; expected immune(0) or dead(3)"
                        )
                        run_length = 0



if __name__ == "__main__":
    unittest.main(verbosity=2)
