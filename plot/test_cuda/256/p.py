import os
import re
import shutil
from collections import defaultdict

# Create destination directories
dest_dirs = {
    0: "density_01",
    1: "density_05",
    2: "density_09"
}
for dir_name in dest_dirs.values():
    os.makedirs(dir_name, exist_ok=True)

# Match pattern: results_W<width>_NP<population>
pattern = re.compile(r"results_W(\d+)_NP(\d+)")

# Collect folders by W value
groups = defaultdict(list)

for folder in os.listdir():
    match = pattern.match(folder)
    if match and os.path.isdir(folder):
        W = int(match.group(1))
        NP = int(match.group(2))
        groups[W].append((NP, folder))

# Process each group
for W, entries in groups.items():
    if len(entries) != 3:
        print(f"⚠️ Skipping W={W}: expected 3 NP values but found {len(entries)}")
        continue

    # Sort by NP
    entries.sort()
    for i, (_, folder) in enumerate(entries):
        target_dir = dest_dirs[i]
        shutil.move(folder, os.path.join(target_dir, folder))
        print(f"✅ Moved {folder} to {target_dir}")
