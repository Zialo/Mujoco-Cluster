import metaworld
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
import tqdm
import os
import random
import shutil
import sys
from PIL import Image
import imageio as iio
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

ITERS = sys.argv[1] if len(sys.argv) > 1 else 10

if not os.path.exists("Imagenes"):
    os.makedirs("Imagenes/")
    os.makedirs("Imagenes/Gripper")
    os.makedirs("Imagenes/Corner")
    os.makedirs("Imagenes/Corner2")
    os.makedirs("Imagenes/Corner3")
    os.makedirs("Imagenes/Top")
    os.makedirs("Imagenes/BehindGripper")

img_gripperPOV_List = []
img_corner_List = []
img_corner2_List = []
img_corner3_List = []
img_topview_List = []
img_behindGripper_List = []
action_list = []

TRAIN_SIZE = round(int(ITERS) * 0.7) if len(sys.argv) == 2 else round(
    int(ITERS) * float(sys.argv[2]))  # Por defecto el TRAIN_SIZE es un 70% del total

iter_list = np.arange(int(ITERS))
train_dataset = random.sample(sorted(iter_list), int(TRAIN_SIZE))
test_dataset = sorted(list(set(iter_list) - set(train_dataset)))

print("Datos:\n - Iteraciones:", str(ITERS), "\n - Train Size:", str(TRAIN_SIZE))
print(train_dataset)
print(test_dataset)

for prob in range(int(ITERS)):
    print("ITERACION - ", str(prob + 1), "/", str(ITERS))
    SEED = random.randint(1, 10000)
    print('Seed: ', str(SEED))
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['pick-place-v2-goal-observable'](seed=SEED)
    policy = SawyerPickPlaceV2Policy()
    obs = env.reset()  # Reset environment
    obs_data = None
    with iio.get_writer("Imagenes/data_" + str(prob) + "_.gif", mode="I", fps=30) as writer:
        for i in tqdm.tqdm(range(25)):
            # a = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            a = policy.get_action(obs)
            action_list.append(a)
            obs, reward, done, info = env.step(a) # Step the environment with the sampled random action
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
            writer.append_data(img_corner)

            # Save for the Images
            img_gripperPOV_List.append(img_gripperPOV)
            img_corner_List.append(img_corner)
            img_corner2_List.append(img_corner2)
            img_corner3_List.append(img_corner3)
            img_topview_List.append(img_topview)
            img_behindGripper_List.append(img_behindGripper)

            if done:
                break

        # plt.imshow(img)
        # plt.savefig(f"img_{i:05d}.jpg")
        # plt.close()

        np.save("Imagenes/positions_" + str(prob) + ".npy", obs_data)
        for i in range(6):
            plt.plot(range(obs_data.shape[0]), obs_data[:, i], label=str(i))
        plt.legend()
        plt.savefig("Imagenes/position" + str(prob) + "_over_time.jpg")
        plt.close()

print(obs_data.shape)

with open('Imagenes/Actions.csv', 'w') as f:
    write = csv.writer(f)
    for i_a in range(len(action_list)):
        row = []
        for a in range(3):
            row.append(action_list[i_a][a])
        write.writerow(row)

for idx in range(len(img_gripperPOV_List)):
    img = Image.fromarray(img_gripperPOV_List[idx])
    img.save(os.path.join("Imagenes/Gripper", f"{idx}.png"))

    img = Image.fromarray(img_corner_List[idx])
    img.save(os.path.join("Imagenes/Corner", f"{idx}.png"))

    img = Image.fromarray(img_corner2_List[idx])
    img.save(os.path.join("Imagenes/Corner2", f"{idx}.png"))

    img = Image.fromarray(img_corner3_List[idx])
    img.save(os.path.join("Imagenes/Corner3", f"{idx}.png"))

    img = Image.fromarray(img_topview_List[idx])
    img.save(os.path.join("Imagenes/Top", f"{idx}.png"))

    img = Image.fromarray(img_behindGripper_List[idx])
    img.save(os.path.join("Imagenes/BehindGripper", f"{idx}.png"))

# Generamos ahora las particiones de Train y Test

train_arange = []
test_arange = []

# Preparo carpeta de Train
if not os.path.exists("Train_dataset"):
    os.makedirs("Train_dataset")
    os.makedirs("Train_dataset/Gripper")
    os.makedirs("Train_dataset/Corner")
    os.makedirs("Train_dataset/Corner2")
    os.makedirs("Train_dataset/Corner3")
    os.makedirs("Train_dataset/Top")
    os.makedirs("Train_dataset/BehindGripper")

else:
    shutil.rmtree("Train_dataset")
    os.makedirs("Train_dataset")
    os.makedirs("Train_dataset/Gripper")
    os.makedirs("Train_dataset/Corner")
    os.makedirs("Train_dataset/Corner2")
    os.makedirs("Train_dataset/Corner3")
    os.makedirs("Train_dataset/Top")
    os.makedirs("Train_dataset/BehindGripper")

# Preparo carpeta de Test
if not os.path.exists("Test_dataset"):
    os.makedirs("Test_dataset")
    os.makedirs("Test_dataset/Gripper")
    os.makedirs("Test_dataset/Corner")
    os.makedirs("Test_dataset/Corner2")
    os.makedirs("Test_dataset/Corner3")
    os.makedirs("Test_dataset/Top")
    os.makedirs("Test_dataset/BehindGripper")

else:
    shutil.rmtree("Test_dataset")
    os.makedirs("Test_dataset")
    os.makedirs("Test_dataset/Gripper")
    os.makedirs("Test_dataset/Corner")
    os.makedirs("Test_dataset/Corner2")
    os.makedirs("Test_dataset/Corner3")
    os.makedirs("Test_dataset/Top")
    os.makedirs("Test_dataset/BehindGripper")

# Imagenes y Archivo de Acciones para Train
file_counter = 0
for i in train_dataset:
    path, dirs, files = next(os.walk("Imagenes"))

    if i == 0:
        num_prueba = i
    else:
        num_prueba = i * 25

    # Imagenes
    for j in range(num_prueba, num_prueba + 25):
        train_arange.append(j)

        # Gripper
        old_image = "Imagenes/Gripper/" + str(j) + ".png"
        new_image = "Train_dataset/Gripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner
        old_image = "Imagenes/Corner/" + str(j) + ".png"
        new_image = "Train_dataset/Corner/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner2
        old_image = "Imagenes/Corner2/" + str(j) + ".png"
        new_image = "Train_dataset/Corner2/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner3
        old_image = "Imagenes/Corner3/" + str(j) + ".png"
        new_image = "Train_dataset/Corner3/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Top
        old_image = "Imagenes/Top/" + str(j) + ".png"
        new_image = "Train_dataset/Top/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # BehindGripper
        old_image = "Imagenes/BehindGripper/" + str(j) + ".png"
        new_image = "Train_dataset/BehindGripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        file_counter += 1

# CSV para Train
# Extraigo las Filas de Train
action_train = []
with open('Imagenes/Actions.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count in train_arange:
            action_train.append(row)
        line_count += 1
    print(f'Processed {line_count} lines.')

# Genero un nuevo CSV de Train
with open('Train_dataset/Train_Actions.csv', 'w') as f:
    write = csv.writer(f)
    for i_a in range(len(action_train)):
        row = []
        for a in range(4):
            row.append(action_train[i_a][a])
        write.writerow(row)

# Imagenes y Archivo de Acciones para Test
file_counter = 0
for i in test_dataset:
    path, dirs, files = next(os.walk("Imagenes"))

    if i == 0:
        num_prueba = i
    else:
        num_prueba = i * 25

    # Imagenes
    for j in range(num_prueba, num_prueba + 25):
        test_arange.append(j)

        # Gripper
        old_image = "Imagenes/Gripper/" + str(j) + ".png"
        new_image = "Test_dataset/Gripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner
        old_image = "Imagenes/Corner/" + str(j) + ".png"
        new_image = "Test_dataset/Corner/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner2
        old_image = "Imagenes/Corner2/" + str(j) + ".png"
        new_image = "Test_dataset/Corner2/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Corner3
        old_image = "Imagenes/Corner3/" + str(j) + ".png"
        new_image = "Test_dataset/Corner3/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # Top
        old_image = "Imagenes/Top/" + str(j) + ".png"
        new_image = "Test_dataset/Top/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        # BehindGripper
        old_image = "Imagenes/BehindGripper/" + str(j) + ".png"
        new_image = "Test_dataset/BehindGripper/" + str(file_counter) + ".png"
        # shutil.copyfile(old_image, new_image)
        shutil.move(old_image, new_image)  # Para mover en lugar de copiar

        file_counter += 1

# CSV para Test
# Extraigo las Filas de Test
action_test = []
with open('Imagenes/Actions.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count in test_arange:
            action_test.append(row)
        line_count += 1
    print(f'Processed {line_count} lines.')

# Genero un nuevo CSV de Test
with open('Test_dataset/Test_Actions.csv', 'w') as f:
    write = csv.writer(f)
    for i_a in range(len(action_test)):
        row = []
        for a in range(4):
            row.append(action_test[i_a][a])
        write.writerow(row)

shutil.rmtree("Imagenes")
