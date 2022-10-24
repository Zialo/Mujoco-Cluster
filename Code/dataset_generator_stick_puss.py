import metaworld
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from metaworld.policies.sawyer_stick_push_v2_policy import SawyerStickPushV2Policy
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

ITERS = sys.argv[1] if len(sys.argv) > 1 else 10
ITS = int(sys.argv[2]) if len(sys.argv) > 1 else 50
NUM = str(sys.argv[4]) if len(sys.argv) > 3 else str(0)
NAME = 'Imagenes_' + NUM

if not os.path.exists(NAME):
    os.makedirs(NAME + "/")
    os.makedirs(NAME + "/Gripper")
    os.makedirs(NAME + "/Corner")
    os.makedirs(NAME + "/Corner2")
    os.makedirs(NAME + "/Corner3")
    os.makedirs(NAME + "/Top")
    os.makedirs(NAME + "/BehindGripper")

img_gripperPOV_List = []
img_corner_List = []
img_corner2_List = []
img_corner3_List = []
img_topview_List = []
img_behindGripper_List = []
action_list = []

TRAIN_SIZE = round(int(ITERS) * 0.7) if len(sys.argv) == 2 else round(
    int(ITERS) * float(sys.argv[3]))  # Por defecto el TRAIN_SIZE es un 70% del total

iter_list = np.arange(int(ITERS))
train_dataset = random.sample(sorted(iter_list), int(TRAIN_SIZE))
test_dataset = sorted(list(set(iter_list) - set(train_dataset)))

train_dataset.sort()
test_dataset.sort()
print("Datos:\n - Intentos:", str(ITERS), "\n - Iteraciones:", str(ITS), "\n - Train Size:", str(TRAIN_SIZE))
print(train_dataset)
print(test_dataset)

lista_FALLO = []
FALLO = 0
pruebas = -1
pruebas_test = 0
pruebas_train = 0
lista_pruebas_total = [-1]
lista_pruebas_train = []
lista_pruebas_test = []
lista_pruebas_train_ind = []
lista_pruebas_test_ind = []

for prob in range(int(ITERS)):
    print("ITERACION - ", str(prob + 1), "/", str(ITERS))
    SEED = random.randint(1, 100000)
    print('Seed: ', str(SEED))
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['stick-push-v2-goal-observable'](seed=SEED)
    policy = SawyerStickPushV2Policy()
    obs = env.reset()  # Reset environment
    obs_data = None
    with iio.get_writer(NAME + "/data_" + str(prob) + "_.gif", mode="I", fps=30) as writer:
        cont_near_object = 0
        for i in tqdm.tqdm(range(ITS)):
            if prob in train_dataset:
                pruebas_train += 1
            else:
                pruebas_test += 1
            pruebas += 1
            # a = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            a = policy.get_action(obs)
            action_list.append(a)
            obs, reward, done, info = env.step(a)  # Step the environment with the sampled random action
            print(info)
            near_object = int(info['in_place_reward'])
            if near_object == 1:
                cont_near_object += 1
            print('PRUEBA: ', str(pruebas), 'in_place_reward: ', str(near_object), 'Cont_Near_Object: ',str(cont_near_object), "\n")
            # obs[-3:] = np.array([0.1, 0.8, 0.2])
            if obs_data is None:
                obs_data = np.concatenate((obs[:3], obs[-3:]), axis=0)
                obs_data = obs_data[np.newaxis, :]
            else:
                cur_obs = np.concatenate((obs[:3], obs[-3:]), axis=0)
                obs_data = np.concatenate((obs_data, cur_obs[np.newaxis, :]), axis=0)

            img_gripperPOV = env.render(offscreen=True, camera_name='gripperPOV', resolution=(128, 128))
            img_corner = env.render(offscreen=True, camera_name='corner', resolution=(128, 128))
            img_corner2 = env.render(offscreen=True, camera_name='corner2', resolution=(128, 128))
            img_corner3 = env.render(offscreen=True, camera_name='corner3', resolution=(128, 128))
            img_topview = env.render(offscreen=True, camera_name='topview', resolution=(128, 128))
            img_behindGripper = env.render(offscreen=True, camera_name='behindGripper', resolution=(128, 128))

            # Save for the GIFs
            # writer.append_data(img_corner)
            # Remove Gifs
            for filename in Path(NAME).glob("*.gif"):
                filename.unlink()
            for filename in Path(NAME).glob("*.jpg"):
                filename.unlink()

            # Save for the Images
            img_gripperPOV_List.append(img_gripperPOV)
            img_corner_List.append(img_corner)
            img_corner2_List.append(img_corner2)
            img_corner3_List.append(img_corner3)
            img_topview_List.append(img_topview)
            img_behindGripper_List.append(img_behindGripper)

            if done:
                break

            if cont_near_object >= 5:
                print('Tope NEAR')
                if prob in train_dataset:
                    print('Train')
                    lista_pruebas_train.append(pruebas)
                    lista_pruebas_train_ind.append(pruebas_train)
                    print(lista_pruebas_train)
                else:
                    print('Test')
                    lista_pruebas_test.append(pruebas)
                    lista_pruebas_test_ind.append(pruebas_test)
                    print(lista_pruebas_test)
                break
            elif (i+1) == ITS:
                print('Tope ITS')
                FALLO += 1
                lista_FALLO.append(prob + 1)
                pruebas -= 1
                if prob in train_dataset:
                    print('Train')
                    lista_pruebas_train.append(pruebas)
                    lista_pruebas_train_ind.append(pruebas_train)
                    print(lista_pruebas_train)
                else:
                    print('Test')
                    lista_pruebas_test.append(pruebas)
                    lista_pruebas_test_ind.append(pruebas_test)
                    print(lista_pruebas_test)
                break	
                
            if FALLO > 0:
            	print('Pruebas con FALLO de Rango: ', str(FALLO))
            	print(lista_FALLO)

        # plt.imshow(img)
        # plt.savefig(f"img_{i:05d}.jpg")
        # plt.close()
        '''
        np.save(NAME + "/positions_" + str(prob) + ".npy", obs_data)
        for i in range(6):
            plt.plot(range(obs_data.shape[0]), obs_data[:, i], label=str(i))
        plt.legend()
        plt.savefig(NAME + "/position" + str(prob) + "_over_time.jpg")
        plt.close()'''

print(obs_data.shape)

lista_pruebas_total += lista_pruebas_train + lista_pruebas_test
lista_pruebas_total.sort()
print('Listas de pruebas')
print(' - Pruebas Train: ', lista_pruebas_train, ' Length: ', str(len(lista_pruebas_train)))
print(' - Pruebas Test: ', lista_pruebas_test, ' Length: ', str(len(lista_pruebas_test)))
print(' - Pruebas Total: ', lista_pruebas_total, ' Length: ', str(len(lista_pruebas_total)))

with open(NAME + '/Actions.csv', 'w') as f:
    write = csv.writer(f)
    for i_a in range(len(action_list)):
        row = []
        for a in range(4):
            row.append(action_list[i_a][a])
        write.writerow(row)
print(NAME)
for idx in range(len(img_gripperPOV_List)):
    img = Image.fromarray(img_gripperPOV_List[idx])
    img.save(os.path.join(NAME + "/Gripper", f"{idx}.png"))

    img = Image.fromarray(img_corner_List[idx])
    img.save(os.path.join(NAME + "/Corner", f"{idx}.png"))

    img = Image.fromarray(img_corner2_List[idx])
    img.save(os.path.join(NAME + "/Corner2", f"{idx}.png"))

    img = Image.fromarray(img_corner3_List[idx])
    img.save(os.path.join(NAME + "/Corner3", f"{idx}.png"))

    img = Image.fromarray(img_topview_List[idx])
    img.save(os.path.join(NAME + "/Top", f"{idx}.png"))

    img = Image.fromarray(img_behindGripper_List[idx])
    img.save(os.path.join(NAME + "/BehindGripper", f"{idx}.png"))

# Generamos ahora las particiones de Train y Test

train_arange = []
test_arange = []

# Preparo carpeta de Train
if not os.path.exists("Train_dataset" + NUM):
    os.makedirs("Train_dataset" + NUM)
    os.makedirs("Train_dataset" + NUM + "/Gripper")
    os.makedirs("Train_dataset" + NUM + "/Corner")
    os.makedirs("Train_dataset" + NUM + "/Corner2")
    os.makedirs("Train_dataset" + NUM + "/Corner3")
    os.makedirs("Train_dataset" + NUM + "/Top")
    os.makedirs("Train_dataset" + NUM + "/BehindGripper")

else:
    shutil.rmtree("Train_dataset" + NUM)
    os.makedirs("Train_dataset" + NUM)
    os.makedirs("Train_dataset" + NUM + "/Gripper")
    os.makedirs("Train_dataset" + NUM + "/Corner")
    os.makedirs("Train_dataset" + NUM + "/Corner2")
    os.makedirs("Train_dataset" + NUM + "/Corner3")
    os.makedirs("Train_dataset" + NUM + "/Top")
    os.makedirs("Train_dataset" + NUM + "/BehindGripper")

# Preparo carpeta de Test
if not os.path.exists("Test_dataset" + NUM):
    os.makedirs("Test_dataset" + NUM)
    os.makedirs("Test_dataset" + NUM + "/Gripper")
    os.makedirs("Test_dataset" + NUM + "/Corner")
    os.makedirs("Test_dataset" + NUM + "/Corner2")
    os.makedirs("Test_dataset" + NUM + "/Corner3")
    os.makedirs("Test_dataset" + NUM + "/Top")
    os.makedirs("Test_dataset" + NUM + "/BehindGripper")

else:
    shutil.rmtree("Test_dataset" + NUM)
    os.makedirs("Test_dataset" + NUM)
    os.makedirs("Test_dataset" + NUM + "/Gripper")
    os.makedirs("Test_dataset" + NUM + "/Corner")
    os.makedirs("Test_dataset" + NUM + "/Corner2")
    os.makedirs("Test_dataset" + NUM + "/Corner3")
    os.makedirs("Test_dataset" + NUM + "/Top")
    os.makedirs("Test_dataset" + NUM + "/BehindGripper")

# Imagenes y Archivo de Acciones para Train
file_counter = 0
print('Train Dataset: ', train_dataset)
for i in range(len(train_dataset)):
    print('Entrada Train: ', str(i))
    path, dirs, files = next(os.walk(NAME))

    # Cogemos el primer valor
    if file_counter == 0 and lista_pruebas_train[0] < lista_pruebas_test[0]:
        num_prueba_inicio = -1
    elif (file_counter == 0 and lista_pruebas_train[0] > lista_pruebas_test[0]) or (file_counter != 0):
        num_prueba_inicio = lista_pruebas_total[lista_pruebas_total.index(lista_pruebas_train[i]) - 1]
    else:
        print('ERROR 251')

    # Cogemos el segundo valor
    if lista_pruebas_total[-1] == num_prueba_inicio:
        break
    num_prueba_final = lista_pruebas_total[lista_pruebas_total.index(num_prueba_inicio) + 1]

    # Imagenes
    print('Train')
    print('Prueba Inicio Train: ', str(num_prueba_inicio+1), 'Prueba Final Train: ', str(num_prueba_final))
    for j in range(num_prueba_inicio+1, num_prueba_final+1):
        train_arange.append(j)

        # Gripper
        old_image = NAME + "/Gripper/" + str(j) + ".png"
        new_image = "Train_dataset" + NUM + "/Gripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner
        old_image = NAME + "/Corner/" + str(j) + ".png"
        new_image = "Train_dataset" + NUM + "/Corner/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner2
        old_image = NAME + "/Corner2/" + str(j) + ".png"
        new_image = "Train_dataset" + NUM + "/Corner2/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner3
        old_image = NAME + "/Corner3/" + str(j) + ".png"
        new_image = "Train_dataset" + NUM + "/Corner3/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Top
        old_image = NAME + "/Top/" + str(j) + ".png"
        new_image = "Train_dataset" + NUM + "/Top/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # BehindGripper
        old_image = NAME + "/BehindGripper/" + str(j) + ".png"
        new_image = "Train_dataset" + NUM + "/BehindGripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        file_counter += 1

# CSV para Train
# Extraigo las Filas de Train
action_train = []
with open(NAME + '/Actions.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count in train_arange:
            action_train.append(row)
        line_count += 1
    print(f'Processed {line_count} lines.')

# Genero un nuevo CSV de Train
with open('Train_dataset' + NUM + '/Train_Actions.csv', 'w') as f:
    write = csv.writer(f)
    for i_a in range(len(action_train)):
        row = []
        for a in range(4):
            row.append(action_train[i_a][a])
        write.writerow(row)

# Imagenes y Archivo de Acciones para Test
file_counter = 0
print('Test dataset: ', test_dataset)
for i in range(len(test_dataset)):
    path, dirs, files = next(os.walk(NAME))
    print('Entrada Test: ', str(i))

    # Cogemos el primer valor
    if file_counter == 0 and lista_pruebas_train[0] > lista_pruebas_test[0]:
        num_prueba_inicio = -1
    elif (file_counter == 0 and lista_pruebas_train[0] < lista_pruebas_test[0]) or (file_counter != 0):
        num_prueba_inicio = lista_pruebas_total[lista_pruebas_total.index(lista_pruebas_test[i]) - 1]
    else:
        print('Error en la linea 236')

    # Cogemos el segundo valor
    if lista_pruebas_total[-1] == num_prueba_inicio:
        break
    num_prueba_final = lista_pruebas_total[lista_pruebas_total.index(num_prueba_inicio) + 1]

    # Imagenes
    print('Test')
    print('Prueba Inicio Test: ', str(num_prueba_inicio+1), 'Prueba Final Test: ', str(num_prueba_final))
    for j in range(num_prueba_inicio+1, num_prueba_final+1):
        test_arange.append(j)

        # Gripper
        old_image = NAME + "/Gripper/" + str(j) + ".png"
        new_image = "Test_dataset" + NUM + "/Gripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner
        old_image = NAME + "/Corner/" + str(j) + ".png"
        new_image = "Test_dataset" + NUM + "/Corner/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner2
        old_image = NAME + "/Corner2/" + str(j) + ".png"
        new_image = "Test_dataset" + NUM + "/Corner2/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner3
        old_image = NAME + "/Corner3/" + str(j) + ".png"
        new_image = "Test_dataset" + NUM + "/Corner3/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Top
        old_image = NAME + "/Top/" + str(j) + ".png"
        new_image = "Test_dataset" + NUM + "/Top/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # BehindGripper
        old_image = NAME + "/BehindGripper/" + str(j) + ".png"
        new_image = "Test_dataset" + NUM + "/BehindGripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        file_counter += 1

# CSV para Test
# Extraigo las Filas de Test
action_test = []
with open(NAME + '/Actions.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count in test_arange:
            action_test.append(row)
        line_count += 1
    print(f'Processed {line_count} lines.')

# Genero un nuevo CSV de Test
with open('Test_dataset' + NUM + '/Test_Actions.csv', 'w') as f:
    write = csv.writer(f)
    for i_a in range(len(action_test)):
        row = []
        for a in range(4):
            row.append(action_test[i_a][a])
        write.writerow(row)

df = pd.DataFrame(lista_pruebas_train_ind)
df.to_csv('Train_dataset' + NUM + '/Tam_pruebas_train.csv', index=False)

df = pd.DataFrame(lista_pruebas_test_ind)
df.to_csv('Test_dataset' + NUM + '/Tam_pruebas_test.csv', index=False)

shutil.rmtree(NAME)

print('Pruebas con FALLO de Rango: ', str(FALLO))
