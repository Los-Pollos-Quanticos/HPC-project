import struct
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
from collections import defaultdict

# Constants
STATE_COLORS = {
    0: 'blue',     # IMMUNE
    1: 'red',      # INFECTED
    2: 'green',    # SUSCEPTIBLE
    3: 'black',    # DEAD
}

DATA_FOLDER = "./report"
PERSON_STRUCT_FORMAT = "iii"  # x, y, state as ints
INT_SIZE = struct.calcsize("i")
PERSON_SIZE = struct.calcsize(PERSON_STRUCT_FORMAT)

CELL_PADDING = 0.1  # spacing inside cell for multiple people

# ─────────────── Simulation constants ───────────────
W            = 15             # grid width
H            = 15             # grid height
# ─────────────────────────────────────────────────────

def load_day_data(day):
    filename = os.path.join(DATA_FOLDER, f"day_{day:03d}.dat")
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        np_value = struct.unpack("i", f.read(INT_SIZE))[0]
        people = []
        for _ in range(np_value):
            x, y, state = struct.unpack(PERSON_STRUCT_FORMAT, f.read(PERSON_SIZE))
            people.append((x, y, state))
    return people

def plot_population(people, day, ax):
    ax.clear()
    ax.set_title(f"Day {day}")
    ax.set_aspect('equal')

    # Prepare per-cell aggregation
    cell_people = defaultdict(list)
    dead_people = []

    state_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for x, y, state in people:
        state_counts[state] += 1
        if state == 3:
            dead_people.append((x, y, state))
        else:
            cell_people[(x, y)].append(state)

    # Plot people in grid cells
    for (x, y), states in cell_people.items():
        n = len(states)
        for i, state in enumerate(states):
            offset_x = ((i % 3) - 1) * CELL_PADDING
            offset_y = ((i // 3) - 0.5) * CELL_PADDING
            ax.plot(x + 0.5 + offset_x, y + 0.5 + offset_y, 'o',
                    color=STATE_COLORS.get(state, 'gray'),
                    markersize=8)

    for i, (x, y, state) in enumerate(dead_people):
        offset_y = i * 0.5
        ax.plot(-1.5, offset_y, 'x',
                color=STATE_COLORS.get(state, 'gray'),
                markersize=10, markeredgewidth=2)

    ax.set_xlim(-3, W + 1)
    ax.set_ylim(-1, H + 2)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True, which='both', color='black', linewidth=0.5)

    legend_text = (
        f"Susceptible: {state_counts[2]}\n"
        f"Infected:    {state_counts[1]}\n"
        f"Immune:      {state_counts[0]}\n"
        f"Dead:        {state_counts[3]}"
    )
    ax.text(W + 1.5, H + 1, legend_text,
            va='top', ha='left', fontsize=10)

    ax.figure.canvas.draw()


class Viewer:
    def __init__(self):
        self.day = 0
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.people = load_day_data(self.day)
        plot_population(self.people, self.day, self.ax)

        axprev = plt.axes([0.1, 0.05, 0.15, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_day)

        axnext = plt.axes([0.3, 0.05, 0.15, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_day)

        plt.show()

    def prev_day(self, event):
        if self.day > 0:
            self.day -= 1
            self.update()

    def next_day(self, event):
        self.day += 1
        self.update()

    def update(self):
        new_people = load_day_data(self.day)
        if new_people is not None:
            self.people = new_people
            plot_population(self.people, self.day, self.ax)
        else:
            self.day -= 1  # Roll back if no data

Viewer()
