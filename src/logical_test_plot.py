#!/usr/bin/env python3
import os
import struct
import matplotlib.pyplot as plt

# ========= CONFIGURATION (edit this path to point at your .dat files) =========
DATA_DIR   = "./report" 
# ==============================================================================

IMMUNE     = 0
INFECTED   = 1
SUSCEPTIBLE= 2
DEAD       = 3

INT_SIZE   = struct.calcsize("i")
PERSON_FMT = "iii"                      
PERSON_SZ  = struct.calcsize(PERSON_FMT)

def load_day_counts(day: int):
    filename = os.path.join(DATA_DIR, f"day_{day:03d}.dat")
    if not os.path.isfile(filename):
        return None, None

    with open(filename, "rb") as f:
        data = f.read(INT_SIZE)
        if len(data) < INT_SIZE:
            return None, None
        NP = struct.unpack("i", data)[0]

        counts = {IMMUNE:0, INFECTED:0, SUSCEPTIBLE:0, DEAD:0}

        for _ in range(NP):
            record = f.read(PERSON_SZ)
            if len(record) < PERSON_SZ:
                break
            x, y, state = struct.unpack(PERSON_FMT, record)
            if state in counts:
                counts[state] += 1
            else:
                pass

    return NP, counts

def gather_all_days():
    days = []
    for fn in os.listdir(DATA_DIR):
        if fn.startswith("day_") and fn.endswith(".dat"):
            try:
                daynum = int(fn.split("_")[1].split(".")[0])
                days.append(daynum)
            except:
                pass
    return sorted(days)

def main():
    days = gather_all_days()
    if len(days) == 0:
        print(f"No day_XXX.dat files found in {DATA_DIR}.")
        return

    sus_percent  = []
    inf_percent  = []
    imm_percent  = []
    dead_percent = []
    X            = []

    for d in days:
        NP, counts = load_day_counts(d)
        if NP is None:
            continue

        total = float(NP)
        s_pct  = 100.0 * (counts[SUSCEPTIBLE] / total)
        i_pct  = 100.0 * (counts[INFECTED]    / total)
        m_pct  = 100.0 * (counts[IMMUNE]      / total)
        d_pct  = 100.0 * (counts[DEAD]        / total)

        X.append(d)
        sus_percent.append(s_pct)
        inf_percent.append(i_pct)
        imm_percent.append(m_pct)
        dead_percent.append(d_pct)

    fig, ax = plt.subplots(figsize=(10, 6))

    pastel_colors = ["#ef233c", "#edf2f4", "#219ebc", "#2b2d42"] 

    ax.stackplot(
        X,
        inf_percent,
        sus_percent,
        imm_percent,
        dead_percent,
        colors=pastel_colors,
        labels=["Infected", "Susceptible", "Immune", "Dead"],
        alpha=0.85
    )

    ax.set_xticks(X)

    ax.set_title("Population State Over Time", fontsize=14, weight="bold")
    ax.set_xlabel("Day",    fontsize=12)
    ax.set_ylabel("Percentage of Population (%)", fontsize=12)
    ax.set_xlim(min(X), max(X))
    ax.set_ylim(0, 100)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
        title="State",
        title_fontsize=11
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
