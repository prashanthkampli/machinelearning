import numpy as np
import csv
import bayespy as bp
from colorama import init

init()

enums = {
    'age': {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4},
    'gender': {'Male': 0, 'Female': 1},
    'familyHistory': {'Yes': 0, 'No': 1},
    'diet': {'High': 0, 'Medium': 1, 'Low': 2},
    'lifeStyle': {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedentary': 3},
    'cholesterol': {'High': 0, 'BorderLine': 1, 'Normal': 2},
    'heartDisease': {'Yes': 0, 'No': 1}
}

with open('heart_disease_data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data = []
    for row in reader:
        row_data = [enums[key][row[key]] for key in enums.keys()]
        data.append(row_data)

data = np.array(data)
N = len(data)

def create_categorical_node(enum_key, num_values):
    p = bp.nodes.Dirichlet(1.0 * np.ones(num_values))
    node = bp.nodes.Categorical(p, plates=(N,))
    node.observe(data[:, list(enums.keys()).index(enum_key)])
    return node, p

nodes = {key: create_categorical_node(key, len(values)) for key, values in enums.items() if key != 'heartDisease'}

plate_sizes = tuple(len(values) for values in enums.values() if len(values) > 1)
p_heartdisease = bp.nodes.Dirichlet(np.ones(2), plates=plate_sizes)
heartdisease = bp.nodes.MultiMixture(list(nodes.values()), bp.nodes.Categorical, p_heartdisease)
heartdisease.observe(data[:, enums['heartDisease']['Yes']])
p_heartdisease.update()

while True:
    inputs = [int(input(f'Enter {k} ({", ".join(v.keys())}): ')) for k, v in enums.items() if k != 'heartDisease']
    inputs_array = np.array([inputs])
    inputs_node = bp.nodes.Categorical(bp.nodes.Dirichlet(np.ones(len(enums)), plates=(1,)), plates=(1,))
    inputs_node.observe(inputs_array)
    prob = heartdisease.get_moments()[0][enums['heartDisease']['Yes']]
    print(f"Probability(HeartDisease) = {prob:.4f}")
    if int(input("Enter 0 to Continue, 1 to Exit: ")) == 1:
        break
