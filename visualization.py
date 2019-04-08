import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def is_clean_data(row):
    if "?" in row:
        return False
    return True


def print_as_table(rows):
    for row in rows:
        print("\t\t".join(map(lambda x: str(x), row)))


def read_csv(path):
    rows = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if is_clean_data(row):
                row = list(map(lambda x: float(x), row))
                rows.append(row)
            else:
                continue
    return rows

def reverse2dArray(array):
    new = []
    for i in range(0,len(array[0])):
        l = []
        for j in range(0,len(array)):
            l.append( array[j][i])
        new.append(l)
    return new


def plot_3():
    valuesMale1 = get_numValuesByAge(filterArray(MaleOnlyArray,2,1.0))
    valuesMale2 = get_numValuesByAge(filterArray(MaleOnlyArray,2,2.0))
    valuesMale3 = get_numValuesByAge(filterArray(MaleOnlyArray,2,3.0))
    valuesMale4 = get_numValuesByAge(filterArray(MaleOnlyArray,2,4.0))

    valuesFemale1 = get_numValuesByAge(filterArray(FemaleOnlyArray,2,1.0))
    valuesFemale2 = get_numValuesByAge(filterArray(FemaleOnlyArray,2,2.0))
    valuesFemale3 = get_numValuesByAge(filterArray(FemaleOnlyArray,2,3.0))
    valuesFemale4 = get_numValuesByAge(filterArray(FemaleOnlyArray,2,4.0))


    plt.bar(ind, valuesMale1,tick_label='type1', width=width , label='Men_typical angin', color=color_green[0], bottom=valuesMale2+valuesMale3+valuesMale4)
    plt.bar(ind, valuesMale2,tick_label='type2', width=width , label='Men_atypical angina',color=color_green[1], bottom=valuesMale3+valuesMale4)
    plt.bar(ind, valuesMale3,tick_label='type3', width=width , label='Men_non-anginal', color=color_green[2], bottom=valuesMale4)
    plt.bar(ind, valuesMale4,tick_label='type4', width=width , label='Men_asymptomatic', color=color_green[3])


    plt.bar(ind+width, valuesFemale1,tick_label='type1', width=width , label='Women_typical angin',  color=color_pink[0],bottom=valuesFemale2+valuesFemale3+valuesFemale4)
    plt.bar(ind+width, valuesFemale2,tick_label='type2', width=width , label='Women_atypical angina', color=color_pink[1], bottom=valuesFemale3+valuesFemale4)
    plt.bar(ind+width, valuesFemale3,tick_label='type3', width=width , label='Women_non-anginal',  color=color_pink[2],bottom=valuesFemale4)
    plt.bar(ind+width, valuesFemale4,tick_label='type4', width=width , label='Women_asymptomatic', color=color_pink[3])


    plt.xticks(ind + width / 2, age_ranges)
    plt.legend(bbox_to_anchor=(-0.1, 0.8))
    plt.title('Relationship Between Chest Pain Type and ages in different sex')


    plt.savefig('static/figures/' + '3' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()