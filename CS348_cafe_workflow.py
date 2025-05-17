
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import *
import random

# Data setup
np.random.seed(42)
random.seed(42)
orders = [f"O{i+1}" for i in range(10)]
tasks = ['Espresso', 'Blending', 'Heating']
resources = ['Espresso_1', 'Espresso_2', 'Blender', 'Oven', 'Barista']
durations = np.random.randint(2, 6, size=(10, 3))
df = pd.DataFrame({
    'Order': np.repeat(orders, 3),
    'Task': tasks * 10,
    'Duration': durations.flatten()
})
resource_map = {
    'Espresso': ['Espresso_1', 'Espresso_2'],
    'Blending': ['Blender'],
    'Heating': ['Oven']
}

# --- MIP ---
model = LpProblem("Cafe_Scheduling", LpMinimize)
assign_vars = LpVariable.dicts("assign", [(i, r) for i in df.index for r in resource_map[df.loc[i, 'Task']]], cat='Binary')
start_vars = LpVariable.dicts("start", df.index, lowBound=0)
makespan = LpVariable("makespan", lowBound=0)
model += makespan
for i in df.index:
    model += lpSum(assign_vars[i, r] for r in resource_map[df.loc[i, 'Task']]) == 1
    model += start_vars[i] + df.loc[i, 'Duration'] <= makespan
M = 1000
for r in resources:
    task_idx = [i for i in df.index if r in resource_map[df.loc[i, 'Task']]]
    for i in task_idx:
        for j in task_idx:
            if i < j:
                model += start_vars[i] + df.loc[i, 'Duration'] <= start_vars[j] + M * (2 - assign_vars[i, r] - assign_vars[j, r])
                model += start_vars[j] + df.loc[j, 'Duration'] <= start_vars[i] + M * (2 - assign_vars[i, r] - assign_vars[j, r])
model.solve()
mip_schedule = pd.DataFrame([
    (df.loc[i, 'Order'], df.loc[i, 'Task'], r, start_vars[i].varValue, df.loc[i, 'Duration'])
    for i in df.index for r in resource_map[df.loc[i, 'Task']] if assign_vars[i, r].varValue == 1
], columns=['Order', 'Task', 'Resource', 'StartTime', 'Duration']).sort_values(by='StartTime')
print("MIP Makespan:", value(makespan))

# --- GA ---
POP_SIZE = 30
N_GENERATIONS = 50
MUTATION_RATE = 0.1
task_options = [resource_map[df.loc[i, 'Task']] for i in df.index]
def create_individual(): return [random.choice(task_options[i]) for i in range(len(df.index))]
def evaluate(ind): 
    timeline = {r: 0 for r in resources}
    starts = []
    for i, r in enumerate(ind):
        start = timeline[r]
        timeline[r] += df.loc[i, 'Duration']
        starts.append(start)
    return max(timeline.values()), starts
def mutate(ind): return [random.choice(task_options[i]) if random.random() < MUTATION_RATE else r for i, r in enumerate(ind)]
def crossover(p1, p2): return p1[:len(p1)//2] + p2[len(p1)//2:]

population = [create_individual() for _ in range(POP_SIZE)]
best_makespan, best_solution, best_start_times = float('inf'), None, None
for _ in range(N_GENERATIONS):
    evaluated = [(evaluate(ind), ind) for ind in population]
    evaluated.sort(key=lambda x: x[0][0])
    population = [ind for (_, ind) in evaluated[:10]]
    if evaluated[0][0][0] < best_makespan:
        best_makespan = evaluated[0][0][0]
        best_solution = evaluated[0][1]
        best_start_times = evaluated[0][0][1]
    while len(population) < POP_SIZE:
        c = mutate(crossover(*random.sample(population, 2)))
        population.append(c)

ga_schedule = pd.DataFrame({
    'Order': df['Order'],
    'Task': df['Task'],
    'Resource': best_solution,
    'StartTime': best_start_times,
    'Duration': df['Duration']
}).sort_values(by='StartTime')
print("GA Makespan:", best_makespan)

# --- Visualization ---
def plot_schedule(schedule, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20.colors
    for idx, row in schedule.iterrows():
        ax.barh(row['Resource'], row['Duration'], left=row['StartTime'], color=colors[idx % len(colors)], edgecolor='black')
        ax.text(row['StartTime'], row['Resource'], f"{row['Order']}-{row['Task']}", va='center', fontsize=8)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Resources")
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_schedule(mip_schedule, "MIP Café Workflow Schedule")
plot_schedule(ga_schedule, "GA Café Workflow Schedule")
