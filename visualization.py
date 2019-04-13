import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



PATH = '/Users/mengmeng/PycharmProjects/COMP9321 Project/data/heart_disease.csv'
age_ranges = ['0-45','45-55','55-65','65-*']
width = 0.35
ind = np.arange(len(age_ranges))

color_green = ['#02C39A','#00A896','#028090','#05668D']
color_pink = ['#9AA09C','#FFCAD4','#F4ACB7','#9D8189']


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



def filterArray(array,index,value):
    new = []
    for i in range(0,len(array)):
        if array[i][index] == value:
            l=[]
            for j in range(0,len(array[0])):
                l.append(array[i][j])
            new.append(l)
    return new

def filterRange(array,index,min,max):
    new = []
    for i in range(0, len(array)):
        if array[i][index] >= min and array[i][index] < max:
            l = []
            for j in range(0, len(array[0])):
                l.append(array[i][j])
            new.append(l)
    return new


def get_numValuesByAge(array):
    values = []
    values.append(len(filterRange(array,0,0,45)))
    values.append(len(filterRange(array,0,45,55)))
    values.append(len(filterRange(array,0,55,65)))
    values.append(len(filterRange(array,0,65,500)))
    return np.array(values)

def toPoints(array,index1,index2):
    reversedArray = reverse2dArray(array)
    new = (reversedArray[index1],reversedArray[index2])
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


def plot_4():
    g1 = toPoints(MaleOnlyArray,0,3)
    g2 = toPoints(FemaleOnlyArray,0,3)


    data = (g1, g2)
    colors = ("green", "red")
    groups = ("Male", "Female")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Relationship Between Resting Blood Pressure and ages in different sex')
    plt.legend(loc=2)
    plt.savefig('static/figures/' + '4' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()


def plot_5():
    g1 = toPoints(MaleOnlyArray, 0, 4)
    g2 = toPoints(FemaleOnlyArray, 0, 4)

    data = (g1, g2)
    colors = ("green", "red")
    groups = ("Male", "Female")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Relationship Between Serum Cholestoral and ages in different sex')
    plt.legend(loc=2)

    plt.savefig('static/figures/' + '5' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

def plot_6():
    valuesMale1 = get_numValuesByAge(filterArray(MaleOnlyArray, 5, 0.0))
    valuesMale2 = get_numValuesByAge(filterArray(MaleOnlyArray, 5, 1.0))

    valuesFemale1 = get_numValuesByAge(filterArray(FemaleOnlyArray, 5, 0.0))
    valuesFemale2 = get_numValuesByAge(filterArray(FemaleOnlyArray, 5, 1.0))


    plt.bar(ind, valuesMale1, tick_label='type1', width=width, label='Men_non_fasting_blood_sugar', color=color_green[0], bottom=valuesMale2)
    plt.bar(ind, valuesMale2, tick_label='type2', width=width, label='Men_fasting_blood_sugar', color=color_green[2])

    plt.bar(ind + width, valuesFemale1, tick_label='type1', width=width, label='Women_non_fasting_blood_sugar angin', color=color_pink[0], bottom=valuesFemale2)
    plt.bar(ind + width, valuesFemale2, tick_label='type2', width=width, label='Women_fasting_blood_sugar', color=color_pink[1])


    plt.xticks(ind + width / 2, age_ranges)
    plt.legend(bbox_to_anchor=(-0.1, 0.8))
    plt.title('Relationship Between Fasting Blood Sugar and ages in different sex')


    plt.savefig('static/figures/' + '6' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

def plot_7():
    valuesMale1 = get_numValuesByAge(filterArray(MaleOnlyArray,6,0.0))
    valuesMale2 = get_numValuesByAge(filterArray(MaleOnlyArray,6,1.0))
    valuesMale3 = get_numValuesByAge(filterArray(MaleOnlyArray,6,2.0))

    valuesFemale1 = get_numValuesByAge(filterArray(FemaleOnlyArray,6,0.0))
    valuesFemale2 = get_numValuesByAge(filterArray(FemaleOnlyArray,6,1.0))
    valuesFemale3 = get_numValuesByAge(filterArray(FemaleOnlyArray,6,2.0))


    plt.bar(ind, valuesMale1,tick_label='type1', width=width , label='Men_normal', color=color_green[0], bottom=valuesMale2+valuesMale3)
    plt.bar(ind, valuesMale2,tick_label='type2', width=width , label='Men_ST-T_abnormal',color=color_green[3], bottom=valuesMale3)
    plt.bar(ind, valuesMale3,tick_label='type3', width=width , label='Men_probable_or_definite_left_ventricular_hypertrophy', color=color_green[1])


    plt.bar(ind+width, valuesFemale1,tick_label='type1', width=width , label='Women_normal',  color=color_pink[0],bottom=valuesFemale2+valuesFemale3)
    plt.bar(ind+width, valuesFemale2,tick_label='type2', width=width , label='Women_ST-T_abnormal', color=color_pink[1], bottom=valuesFemale3)
    plt.bar(ind+width, valuesFemale3,tick_label='type3', width=width , label='Women_probable_or_definite_left_ventricular_hypertrophy',  color=color_pink[2])


    plt.xticks(ind + width / 2, age_ranges)
    plt.legend(bbox_to_anchor=(-0.1, 0.8))
    plt.title('Relationship Between Resting Electrocardiographic Results and ages in different sex')


    plt.savefig('static/figures/' + '7' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()


def plot_8():
    g1 = toPoints(MaleOnlyArray, 0, 7)
    g2 = toPoints(FemaleOnlyArray, 0, 7)

    data = (g1, g2)
    colors = ("green", "red")
    groups = ("Male", "Female")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Relationship Between Maximum Heart Rate Achieved  and ages in different sex')
    plt.xlabel = 'age'
    plt.ylabel = 'maximum heart rate achieved'
    plt.legend(loc=2)

    plt.savefig('static/figures/' + '8' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

def plot_9():
    valuesMale1 = get_numValuesByAge(filterArray(MaleOnlyArray, 8, 0.0))
    valuesMale2 = get_numValuesByAge(filterArray(MaleOnlyArray, 8, 1.0))

    valuesFemale1 = get_numValuesByAge(filterArray(FemaleOnlyArray, 8, 0.0))
    valuesFemale2 = get_numValuesByAge(filterArray(FemaleOnlyArray, 8, 1.0))

    plt.bar(ind, valuesMale1, tick_label='type1', width=width, label='Men_non_induced_angina',
            color=color_green[0], bottom=valuesMale2)
    plt.bar(ind, valuesMale2, tick_label='type2', width=width, label='Men_induced_angina', color=color_green[2])

    plt.bar(ind + width, valuesFemale1, tick_label='type1', width=width, label='Women_non_induced_angina',
            color=color_pink[0], bottom=valuesFemale2)
    plt.bar(ind + width, valuesFemale2, tick_label='type2', width=width, label='Women_induced_angina',
            color=color_pink[1])

    plt.xticks(ind + width / 2, age_ranges)
    plt.legend(bbox_to_anchor=(-0.1, 0.8))
    plt.title('Relationship Between Exercise Induced Angina and ages in different sex')


    plt.savefig('static/figures/' + '9' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

def plot_10():
    g1 = toPoints(MaleOnlyArray, 0, 9)
    g2 = toPoints(FemaleOnlyArray, 0, 9)

    data = (g1, g2)
    colors = ("green", "red")
    groups = ("Male", "Female")

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('Relationship Between Oldpeak and ages in different sex')
    plt.xlabel = 'age'
    plt.ylabel = 'oldpeak'
    plt.legend(loc=2)

    plt.savefig('static/figures/' + '10' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

def plot_11():
    valuesMale1 = get_numValuesByAge(filterArray(MaleOnlyArray,10,1.0))
    valuesMale2 = get_numValuesByAge(filterArray(MaleOnlyArray,10,2.0))
    valuesMale3 = get_numValuesByAge(filterArray(MaleOnlyArray,10,3.0))

    valuesFemale1 = get_numValuesByAge(filterArray(FemaleOnlyArray,10,1.0))
    valuesFemale2 = get_numValuesByAge(filterArray(FemaleOnlyArray,10,2.0))
    valuesFemale3 = get_numValuesByAge(filterArray(FemaleOnlyArray,10,3.0))


    plt.bar(ind, valuesMale1,tick_label='type1', width=width , label='Men_slop=1', color=color_green[0], bottom=valuesMale2+valuesMale3)
    plt.bar(ind, valuesMale2,tick_label='type2', width=width , label='Men_slop=2',color=color_green[3], bottom=valuesMale3)
    plt.bar(ind, valuesMale3,tick_label='type3', width=width , label='Men_slop=3', color=color_green[1])


    plt.bar(ind+width, valuesFemale1,tick_label='type1', width=width , label='Women_slop=1',  color=color_pink[0],bottom=valuesFemale2+valuesFemale3)
    plt.bar(ind+width, valuesFemale2,tick_label='type2', width=width , label='Women_slop=2', color=color_pink[1], bottom=valuesFemale3)
    plt.bar(ind+width, valuesFemale3,tick_label='type3', width=width , label='Women_slop=3',  color=color_pink[2])


    plt.xticks(ind + width / 2, age_ranges)
    plt.legend(bbox_to_anchor=(-0.1, 0.8))
    plt.title('Relationship Between the slope of the peak exercise ST segment and ages in different sex')


    plt.savefig('static/figures/' + '11' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

def plot_12():
    valuesMale1 = get_numValuesByAge(filterArray(MaleOnlyArray, 11, 0.0))
    valuesMale2 = get_numValuesByAge(filterArray(MaleOnlyArray, 11, 1.0))
    valuesMale3 = get_numValuesByAge(filterArray(MaleOnlyArray, 11, 2.0))
    valuesMale4 = get_numValuesByAge(filterArray(MaleOnlyArray, 11, 3.0))

    valuesFemale1 = get_numValuesByAge(filterArray(FemaleOnlyArray, 11, 0.0))
    valuesFemale2 = get_numValuesByAge(filterArray(FemaleOnlyArray, 11, 1.0))
    valuesFemale3 = get_numValuesByAge(filterArray(FemaleOnlyArray, 11, 2.0))
    valuesFemale4 = get_numValuesByAge(filterArray(FemaleOnlyArray, 11, 3.0))

    plt.bar(ind, valuesMale1, tick_label='type1', width=width, label='Men_0_major_vessels', color=color_green[0],
            bottom=valuesMale2 + valuesMale3 + valuesMale4)
    plt.bar(ind, valuesMale2, tick_label='type2', width=width, label='Men_1_major_vessels', color=color_green[1],
            bottom=valuesMale3 + valuesMale4)
    plt.bar(ind, valuesMale3, tick_label='type3', width=width, label='Men_2_major_vessels', color=color_green[2],
            bottom=valuesMale4)
    plt.bar(ind, valuesMale4, tick_label='type4', width=width, label='Men_3_major_vessels', color=color_green[3])

    plt.bar(ind + width, valuesFemale1, tick_label='type1', width=width, label='Women_0_major_vessels',
            color=color_pink[0], bottom=valuesFemale2 + valuesFemale3 + valuesFemale4)
    plt.bar(ind + width, valuesFemale2, tick_label='type2', width=width, label='Women_1_major_vessels',
            color=color_pink[1], bottom=valuesFemale3 + valuesFemale4)
    plt.bar(ind + width, valuesFemale3, tick_label='type3', width=width, label='Women_2_major_vessels', color=color_pink[2],
            bottom=valuesFemale4)
    plt.bar(ind + width, valuesFemale4, tick_label='type4', width=width, label='Women_3_major_vessels',
            color=color_pink[3])

    plt.xticks(ind + width / 2, age_ranges)
    plt.legend(bbox_to_anchor=(-0.1, 0.8))
    plt.title('Relationship Between number of major vessels and ages in different sex')


    plt.savefig('static/figures/' + '12' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

def plot_13():
    valuesMale1 = get_numValuesByAge(filterArray(MaleOnlyArray, 12, 3.0))
    valuesMale2 = get_numValuesByAge(filterArray(MaleOnlyArray, 12, 6.0))
    valuesMale3 = get_numValuesByAge(filterArray(MaleOnlyArray, 12, 7.0))

    valuesFemale1 = get_numValuesByAge(filterArray(FemaleOnlyArray, 12, 3.0))
    valuesFemale2 = get_numValuesByAge(filterArray(FemaleOnlyArray, 12, 6.0))
    valuesFemale3 = get_numValuesByAge(filterArray(FemaleOnlyArray, 12, 7.0))

    plt.bar(ind, valuesMale1, tick_label='type1', width=width, label='Men_normal', color=color_green[0],
            bottom=valuesMale2 + valuesMale3)
    plt.bar(ind, valuesMale2, tick_label='type2', width=width, label='Men_fixed_defect', color=color_green[3],
            bottom=valuesMale3)
    plt.bar(ind, valuesMale3, tick_label='type3', width=width, label='Men_reversable_defect', color=color_green[1])

    plt.bar(ind + width, valuesFemale1, tick_label='type1', width=width, label='Women_normal', color=color_pink[0],
            bottom=valuesFemale2 + valuesFemale3)
    plt.bar(ind + width, valuesFemale2, tick_label='type2', width=width, label='Women_fixed_defect', color=color_pink[1],
            bottom=valuesFemale3)
    plt.bar(ind + width, valuesFemale3, tick_label='type3', width=width, label='Women_reversable_defect', color=color_pink[2])

    plt.xticks(ind + width / 2, age_ranges)
    plt.legend(bbox_to_anchor=(-0.1, 0.8))
    plt.title('Relationship Between thal(Thalassemia) and ages in different sex')

    plt.savefig('static/figures/' + '13' + '.png', fotmat='png', bbox_inches='tight')

    # plt.show()

OriginArray = read_csv(PATH)
MaleOnlyArray = filterArray(OriginArray,1,0.0)
FemaleOnlyArray = filterArray(OriginArray,1,1.0)

print_as_table(OriginArray)

for i in range(3, 14):

    eval("plot_%s" % i)()
    plt.close()
