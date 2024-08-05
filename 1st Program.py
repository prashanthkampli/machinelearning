import csv

with open(r"enjoysport.csv") as csv_file:
    readcsv = csv.reader(csv_file, delimiter=',')
    data = [row for row in readcsv]

headers = data[0]
data = data[1:]

positive_examples = [row for row in data if row[-1].upper() == "YES"]

print("\nThe positive examples are:")
for x in positive_examples:
    print(x)
print("\n")

hypo = positive_examples[0][:-1]

for i in range(1, len(positive_examples)):
    for j in range(len(hypo)):
        if hypo[j] != positive_examples[i][j]:
            hypo[j] = '?'

print("\nThe maximally specific Find-S hypothesis for the given training examples is:")
print(hypo)
