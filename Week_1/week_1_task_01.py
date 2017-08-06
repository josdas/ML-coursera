import numpy as np
import pandas
import math

data = pandas.read_csv('_ea07570741a3ec966e284208f588e50e_titanic.csv', index_col='PassengerId')


def print_to_file(filename, *args):
    file = open(filename, 'w')
    file.write(' '.join(str(v) for v in args))
    file.close()


print_to_file("01", 8, 5, 7)

sex = data['Sex'].tolist()
print_to_file("01", sex.count("male"), sex.count("female"))

survived = data['Survived'].tolist()
percent_survived = sum(survived) / len(survived) * 100
print_to_file("02", round(percent_survived, 2))

pclass = data['Pclass'].tolist()
percent_first_class = pclass.count(1) / len(survived) * 100
print_to_file("03", round(percent_first_class, 2))

age = data['Age'].tolist()
print_to_file("04", round(np.nanmean(age), 2), np.nanmedian(age))

print_to_file("05", round(data['SibSp'].corr(data['Parch']), 2))

name = data["Name"].tolist()
names_count = {}

for i in range(0, len(name)):
    if sex[i] == 'female':
        first_name = name[i].split(' ')[2]
        result = ''
        for c in first_name:
            if c.isalpha():
                result = result + c
        names_count[result] = names_count.get(result, 0) + 1
print(names_count)
max_count = -math.inf
result_name = ''
for v in names_count.items():
    if v[1] > max_count:
        print(v)
        max_count = v[1]
        result_name = v[0]
print_to_file("06", result_name)
