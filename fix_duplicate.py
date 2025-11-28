"""
Fix duplicate assignment in recommender.py
"""

with open('src/models/recommender.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Remove the incorrect assignment
    if "csr_matrix = models['csr_matrix']" in line:
        continue
    new_lines.append(line)

with open('src/models/recommender.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(" Removed incorrect 'csr_matrix' assignment")
