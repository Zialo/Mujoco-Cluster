import metaworld
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from metaworld.policies.sawyer_pick_out_of_hole_v2_policy import SawyerPickOutOfHoleV2Policy
import tqdm
import os
import random
import shutil
import sys
from PIL import Image
import pandas as pd
import imageio as iio
from pathlib import Path
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            
OPTION = int(sys.argv[1]) if len(sys.argv) > 1 else 0

NAME = 'Basketball_depth'
carpetas = ['basketball1',
            'basketball2',
            'basketball3',
            'basketball4']

csv_train = ['basketball1/Train_dataset/Train_Actions.csv',
             'basketball2/Train_dataset/Train_Actions.csv',
             'basketball3/Train_dataset/Train_Actions.csv',
             'basketball4/Train_dataset/Train_Actions.csv']

csv_test =  ['basketball1/Test_dataset/Test_Actions.csv',
             'basketball2/Test_dataset/Test_Actions.csv',
             'basketball3/Test_dataset/Test_Actions.csv',
             'basketball4/Test_dataset/Test_Actions.csv']

csv_tam_train = ['basketball1/Train_dataset/Tam_pruebas_train.csv',
                 'basketball2/Train_dataset/Tam_pruebas_train.csv',
                 'basketball3/Train_dataset/Tam_pruebas_train.csv',
                 'basketball4/Train_dataset/Tam_pruebas_train.csv']

csv_tam_test =  ['basketball1/Test_dataset/Tam_pruebas_test.csv',
                 'basketball2/Test_dataset/Tam_pruebas_test.csv',
                 'basketball3/Test_dataset/Tam_pruebas_test.csv',
                 'basketball4/Test_dataset/Tam_pruebas_test.csv']
                 
csv_train_obs = ['basketball1/Train_dataset/Train_Obs.csv',
                 'basketball2/Train_dataset/Train_Obs.csv',
                 'basketball3/Train_dataset/Train_Obs.csv',
                 'basketball4/Train_dataset/Train_Obs.csv']

csv_test_obs =  ['basketball1/Test_dataset/Test_Obs.csv',
                 'basketball2/Test_dataset/Test_Obs.csv',
                 'basketball3/Test_dataset/Test_Obs.csv',
                 'basketball4/Test_dataset/Test_Obs.csv']

# Genero las carpetas
if not os.path.exists(NAME):
    os.makedirs(NAME + "/")
    os.makedirs(NAME + "/Test_dataset")
    os.makedirs(NAME + "/Test_dataset/Corner2_d")
    os.makedirs(NAME + "/Test_dataset/Top_d")
    os.makedirs(NAME + "/Test_dataset/Corner2")
    os.makedirs(NAME + "/Test_dataset/BehindGripper_d")
    os.makedirs(NAME + "/Test_dataset/Top")
    os.makedirs(NAME + "/Test_dataset/BehindGripper")
    os.makedirs(NAME + "/Train_dataset")
    os.makedirs(NAME + "/Train_dataset/Corner2_d")
    os.makedirs(NAME + "/Train_dataset/Top_d")
    os.makedirs(NAME + "/Train_dataset/Corner2")
    os.makedirs(NAME + "/Train_dataset/BehindGripper_d")
    os.makedirs(NAME + "/Train_dataset/Top")
    os.makedirs(NAME + "/Train_dataset/BehindGripper")
else:
    shutil.rmtree(NAME)
    os.makedirs(NAME + "/")
    os.makedirs(NAME + "/Test_dataset")
    os.makedirs(NAME + "/Test_dataset/Corner2_d")
    os.makedirs(NAME + "/Test_dataset/Top_d")
    os.makedirs(NAME + "/Test_dataset/Corner2")
    os.makedirs(NAME + "/Test_dataset/BehindGripper_d")
    os.makedirs(NAME + "/Test_dataset/Top")
    os.makedirs(NAME + "/Test_dataset/BehindGripper")
    os.makedirs(NAME + "/Train_dataset")
    os.makedirs(NAME + "/Train_dataset/Corner2_d")
    os.makedirs(NAME + "/Train_dataset/Top_d")
    os.makedirs(NAME + "/Train_dataset/Corner2")
    os.makedirs(NAME + "/Train_dataset/BehindGripper_d")
    os.makedirs(NAME + "/Train_dataset/Top")
    os.makedirs(NAME + "/Train_dataset/BehindGripper")

file_counter = 0
file_counter_test = 0

for carpeta in carpetas:

    # Imagenes de Train
    path_train = carpeta + '/Train_dataset/'
    path_aux = path_train + '/Corner2'
    path, dirs, files = next(os.walk(path_aux))
    print(path)
    print(len(files))

    for j in range(len(files)):

        # Corner2
        old_image = path_train + "/Corner2/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Corner2/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # Top
        old_image = path_train + "/Top/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Top/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # BehindGripper
        old_image = path_train + "/BehindGripper/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/BehindGripper/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)
        
        # Corner2_depth
        old_image = path_train + "/Corner2_d/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Corner2_d/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # Top_depth
        old_image = path_train + "/Top_d/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Top_d/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # BehindGripper_depth
        old_image = path_train + "/BehindGripper_d/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/BehindGripper_d/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        file_counter += 1
        if file_counter % 100 == 0:
            print(file_counter)
            print(old_image, " > ", new_image)

    # CSV para Train
    print('Empieza CSV Train')
    action_train = []
    for i in csv_train:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count in range(len(files)):
                    action_train.append(row)

    # Genero un nuevo CSV de Train
    with open(NAME + "/Train_dataset/Train_Actions.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(action_train)):
            row = []
            for a in range(4):
                row.append(action_train[i_a][a])
            write.writerow(row)
            
    # CSV para Train OBS
    print('Empieza CSV Train')
    obs_train = []
    for i in csv_train_obs:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count in range(len(files)):
                    obs_train.append(row)

    # Genero un nuevo CSV de Train
    with open(NAME + "/Train_dataset/Train_Obs.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(obs_train)):
            row = []
            for a in range(4):
                row.append(obs_train[i_a][a])
            write.writerow(row)
            
    with open(NAME + "/Train_dataset/Train_Actions_ng.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(action_train)):
            row = []
            for a in range(3):
                row.append(action_train[i_a][a])
            write.writerow(row)

    # CSV para Tam_Pruebas_Train
    print('Empieza CSV Tam_pruebas_Train')
    tam = []
    tam_max = 0
    for i in csv_tam_train:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if (i == csv_tam_train[0]) or ((i != csv_tam_train[0]) and (int(row[0]) != 0)):  # Para saltarnos el primer 0 del resto
                    value = int(row[0]) + tam_max
                    tam.append(value)
        tam_max = value
        print('Tam_max: ', str(tam_max))
        print('Tam Length: ', str(len(tam)))

    # Genero un nuevo CSV de Tam
    with open(NAME + "/Train_dataset/Tam_pruebas_train.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(tam)):
            row = []
            row.append(tam[i_a])
            write.writerow(row)

    # Imagenes de Test
    path_test = carpeta + '/Test_dataset/'
    path_aux = path_test + '/Corner2'
    path, dirs, files = next(os.walk(path_aux))
    print(path)
    print(len(files))

    for j in range(len(files)):
     
        # Corner2
        old_image = path_test + "/Corner2/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Corner2/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # Top
        old_image = path_test + "/Top/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Top/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # BehindGripper
        old_image = path_test + "/BehindGripper/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/BehindGripper/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)
        
        # Corner2_depth
        old_image = path_test + "/Corner2_d/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Corner2_d/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # Top_depth
        old_image = path_test + "/Top_d/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Top_d/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # BehindGripper_depth
        old_image = path_test + "/BehindGripper_d/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/BehindGripper_d/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        file_counter_test += 1
        if file_counter_test % 100 == 0:
            print(file_counter_test)
            print(old_image, " > ", new_image)

    # CSV para Test
    print('Empieza CSV Test')
    action_test = []
    for i in csv_test:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count in range(len(files)):
                    action_test.append(row)

    # Genero un nuevo CSV de Test
    with open(NAME + "/Test_dataset/Test_Actions.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(action_test)):
            row = []
            for a in range(4):
                row.append(action_test[i_a][a])
            write.writerow(row)
   
    # CSV para Test OBS
    print('Empieza CSV Test')
    obs_test = []
    for i in csv_test_obs:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count in range(len(files)):
                    obs_test.append(row)

    # Genero un nuevo CSV de Test
    with open(NAME + "/Test_dataset/Test_Obs.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(obs_test)):
            row = []
            for a in range(4):
                row.append(obs_test[i_a][a])
            write.writerow(row)
            
    with open(NAME + "/Test_dataset/Test_Actions_ng.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(action_test)):
            row = []
            for a in range(3):
                row.append(action_test[i_a][a])
            write.writerow(row)

    # CSV para Tam_Pruebas_Test
    print('Empieza CSV Tam_pruebas_Test')
    tam = []
    tam_max = 0
    for i in csv_tam_test:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if (i == csv_tam_test[0]) or ((i != csv_tam_test[0]) and (int(row[0]) != 0)):  # Para saltarnos el primer 0 del resto
                    value = int(row[0]) + tam_max
                    tam.append(value)
        tam_max = value
        print('Tam_max: ', str(tam_max))
        print('Tam Length: ', str(len(tam)))

    # Genero un nuevo CSV de Tam
    with open(NAME + "/Test_dataset/Tam_pruebas_test.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(tam)):
            row = []
            row.append(tam[i_a])
            write.writerow(row)
