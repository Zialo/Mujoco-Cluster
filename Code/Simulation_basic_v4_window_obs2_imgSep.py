from memory_profiler import profile
import os
import sys
import csv
import numpy as np
import pandas as pd
import pickle
import shutil
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms
from torch import nn
from absl import app
from absl import flags
import joblib
from sklearn.preprocessing import MinMaxScaler
import metaworld
import random
import matplotlib.pyplot as plt
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
import tqdm
import random
import imageio as iio
import time as timer
from statistics import median
from csv import reader
from csv import writer
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

img_size = [128, 128]
NORM = sys.argv[1] if len(sys.argv) > 1 else 0
NUM_PRUEBAS = sys.argv[2] if len(sys.argv) > 2 else 10
NUM_ITERS = sys.argv[3] if len(sys.argv) > 3 else 100
SAVE = sys.argv[4] if len(sys.argv) > 4 else 1
OPTION = sys.argv[5] if len(sys.argv) > 5 else False
MODEL_NAME = sys.argv[6] if len(sys.argv) > 6 else 'model_pytorch_01593'

if OPTION != True and OPTION != False:
    OPTION = True

print('Parametros introducidos: \n -NORM: ', str(NORM), '\n -NUM_PRUEBAS: ', str(NUM_PRUEBAS), '\n -NUM_ITERS: ', str(NUM_ITERS), '\n -OPTION: ', str(OPTION), '\n -SAVE: ', str(SAVE), '\n -MODEL: ', str(MODEL_NAME), '\n')

# Preparo carpeta de Train
if not os.path.exists("Imagenes GIF imgSepW"):
    os.makedirs("Imagenes GIF imgSepW")
    os.makedirs("Imagenes GIF imgSepW/Top")
    os.makedirs("Imagenes GIF imgSepW/Corner")
    os.makedirs("Imagenes GIF imgSepW/Corner2")
    os.makedirs("Imagenes GIF imgSepW/Corner3")
    os.makedirs("Imagenes GIF imgSepW/Gripper")
    os.makedirs("Imagenes GIF imgSepW/BehindGripper")
    os.makedirs("Imagenes GIF imgSepW/Top_d")
    os.makedirs("Imagenes GIF imgSepW/Corner_d")
    os.makedirs("Imagenes GIF imgSepW/Corner2_d")
    os.makedirs("Imagenes GIF imgSepW/Corner3_d")
    os.makedirs("Imagenes GIF imgSepW/Gripper_d")
    os.makedirs("Imagenes GIF imgSepW/BehindGripper_d")
else:
    shutil.rmtree("Imagenes GIF imgSepW")
    os.makedirs("Imagenes GIF imgSepW")
    os.makedirs("Imagenes GIF imgSepW/Top")
    os.makedirs("Imagenes GIF imgSepW/Corner")
    os.makedirs("Imagenes GIF imgSepW/Corner2")
    os.makedirs("Imagenes GIF imgSepW/Corner3")
    os.makedirs("Imagenes GIF imgSepW/Gripper")
    os.makedirs("Imagenes GIF imgSepW/BehindGripper")
    os.makedirs("Imagenes GIF imgSepW/Top_d")
    os.makedirs("Imagenes GIF imgSepW/Corner_d")
    os.makedirs("Imagenes GIF imgSepW/Corner2_d")
    os.makedirs("Imagenes GIF imgSepW/Corner3_d")
    os.makedirs("Imagenes GIF imgSepW/Gripper_d")
    os.makedirs("Imagenes GIF imgSepW/BehindGripper_d")
# Preparo la carpeta de Resultados
if not os.path.exists("Resultados"):
	os.makedirs("Resultados")


# PREDICT
class MultiImage(nn.Module):
    def __init__(self, fe, clf):
        super(MultiImage, self).__init__()
        self.fe = fe
        self.clf = clf
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x1, x2, x3, x4, x5 = x

        x2 = torch.from_numpy(x2)
        x2 = x2.unsqueeze(0)
        x2 = x2.to(torch.float32)

        x3 = torch.from_numpy(x3)
        x3 = x3.unsqueeze(0)
        x3 = x3.to(torch.float32)

        x4 = torch.from_numpy(x4)
        x4 = x4.unsqueeze(0)
        x4 = x4.to(torch.float32)

        x5 = torch.from_numpy(x5)
        x5 = x5.unsqueeze(0)
        x5 = x5.to(torch.float32)

        f1 = self.fe(x1)
        f = self.flatten(self.avg_pool(f1))

        f = torch.cat((f, x2, x3, x4, x5), dim=1)
        return self.clf(f)

def predict(model, env, env_prev, action_prev, obs_act, obs_prev, obs_prev2, i, SAVE):
    model = model.eval()
    # Get image from observation
    im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12, im1_d, im2_d, im3_d, im4_d, im5_d, im6_d = get_image_from_observation(env, env_prev, i)

    if int(SAVE) == 1:
        # Save three first Images
        name_1 = "Imagenes GIF imgSepW/Top/" + str(i) + ".png"
        name_2 = "Imagenes GIF imgSepW/Corner/" + str(i) + ".png"
        name_3 = "Imagenes GIF imgSepW/Corner2/" + str(i) + ".png"
        name_4 = "Imagenes GIF imgSepW/Corner3/" + str(i) + ".png"
        name_5 = "Imagenes GIF imgSepW/Gripper/" + str(i) + ".png"
        name_6 = "Imagenes GIF imgSepW/BehindGripper/" + str(i) + ".png"

        name_1_d = "Imagenes GIF imgSepW/Top_d/" + str(i) + ".csv"
        name_2_d = "Imagenes GIF imgSepW/Corner_d/" + str(i) + ".csv"
        name_3_d = "Imagenes GIF imgSepW/Corner2_d/" + str(i) + ".csv"
        name_4_d = "Imagenes GIF imgSepW/Corner3_d/" + str(i) + ".csv"
        name_5_d = "Imagenes GIF imgSepW/Gripper_d/" + str(i) + ".csv"
        name_6_d = "Imagenes GIF imgSepW/BehindGripper_d/" + str(i) + ".csv"

        name_1_di = "Imagenes GIF imgSepW/Top_d/" + str(i) + ".png"
        name_2_di = "Imagenes GIF imgSepW/Corner_d/" + str(i) + ".png"
        name_3_di = "Imagenes GIF imgSepW/Corner2_d/" + str(i) + ".png"
        name_4_di = "Imagenes GIF imgSepW/Corner3_d/" + str(i) + ".png"
        name_5_di = "Imagenes GIF imgSepW/Gripper_d/" + str(i) + ".png"
        name_6_di = "Imagenes GIF imgSepW/BehindGripper_d/" + str(i) + ".png"

        top = Image.fromarray(im1)
        corner = Image.fromarray(im2)
        corner2 = Image.fromarray(im3)
        corner3 = Image.fromarray(im4)
        gripper = Image.fromarray(im5)
        behindgripper = Image.fromarray(im6)

        top_d = Image.fromarray(im1_d)
        corner_d = Image.fromarray(im2_d)
        corner2_d = Image.fromarray(im3_d)
        corner3_d = Image.fromarray(im4_d)
        gripper_d = Image.fromarray(im5_d)
        behindgripper_d = Image.fromarray(im6_d)

        top.save(name_1, "PNG")
        corner.save(name_2, "PNG")
        corner2.save(name_3, "PNG")
        corner3.save(name_4, "PNG")
        gripper.save(name_5, "PNG")
        behindgripper.save(name_6, "PNG")

        plt.imsave(name_1_di, im1_d)
        plt.imsave(name_2_di, im2_d)
        plt.imsave(name_3_di, im3_d)
        plt.imsave(name_4_di, im4_d)
        plt.imsave(name_5_di, im5_d)
        plt.imsave(name_6_di, im6_d)

        pd.DataFrame(im1_d).to_csv(name_1_d, index_label = None, header  = None)
        pd.DataFrame(im2_d).to_csv(name_2_d, index_label = None, header  = None)
        pd.DataFrame(im3_d).to_csv(name_3_d, index_label = None, header  = None)
        pd.DataFrame(im4_d).to_csv(name_4_d, index_label = None, header  = None)
        pd.DataFrame(im5_d).to_csv(name_5_d, index_label = None, header  = None)
        pd.DataFrame(im6_d).to_csv(name_6_d, index_label = None, header  = None)

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48318733, 0.37641326, 0.35779971),
                             (0.17344827, 0.2299165,  0.23088937))
    ])

    start_time_pre = timer.time()

    im1 = transformations(im1)
    im2 = transformations(im2)
    im3 = transformations(im3)
    im4 = transformations(im4)
    im5 = transformations(im5)
    im6 = transformations(im6)
    im7 = transformations(im7)
    im8 = transformations(im8)
    im9 = transformations(im9)
    im10 = transformations(im10)
    im11 = transformations(im11)
    im12 = transformations(im12)

    im_actual = torch.cat((im1, im2, im3, im4, im5, im6), axis=0)
    im_previa = torch.cat((im7, im8, im9, im10, im11, im12), axis=0)
    image = torch.cat((im_actual, im_previa), axis=0)
    image = image.unsqueeze(dim=0)

    preprocessing_time.append(timer.time() - start_time_pre)

    with torch.no_grad():
        start_time_mod = timer.time()
        pred = model((image, action_prev, obs_act, obs_prev, obs_prev2))
        model_time.append(timer.time() - start_time_mod)
    pred = pred.detach().cpu().numpy()
    prediction = pred[0, :].tolist()
    #prediction.append(1)
    # print("PRED", prediction, " FIN")
    return prediction

def get_image_from_observation(env, env_prev, i):
    im1, im1_d = env.render(offscreen=False, camera_name='topview', resolution=(img_size[0], img_size[1]), mode ='Depth_array')
    im2, im2_d = env.render(offscreen=False, camera_name='corner', resolution=(img_size[0], img_size[1]), mode ='Depth_array')
    im3, im3_d = env.render(offscreen=False, camera_name='corner2', resolution=(img_size[0], img_size[1]), mode ='Depth_array')
    im4, im4_d = env.render(offscreen=False, camera_name='corner3', resolution=(img_size[0], img_size[1]), mode ='Depth_array')
    im5, im5_d = env.render(offscreen=False, camera_name='gripperPOV', resolution=(img_size[0], img_size[1]), mode ='Depth_array')
    im6, im6_d = env.render(offscreen=False, camera_name='behindGripper', resolution=(img_size[0], img_size[1]), mode ='Depth_array')

    if ((i % int(NUM_ITERS) == 0) or (i == 0)):
        im7 = im1
        im8 = im2
        im9 = im3
        im10 = im4
        im11 = im5
        im12 = im6

    else:
        im7, im7_d = env.render(offscreen=False, camera_name='topview', resolution=(img_size[0], img_size[1]),
                                mode='Depth_array')
        im8, im8_d = env.render(offscreen=False, camera_name='corner', resolution=(img_size[0], img_size[1]),
                                mode='Depth_array')
        im9, im9_d = env.render(offscreen=False, camera_name='corner2', resolution=(img_size[0], img_size[1]),
                                mode='Depth_array')
        im10, im10_d = env.render(offscreen=False, camera_name='corner3', resolution=(img_size[0], img_size[1]),
                                mode='Depth_array')
        im11, im11_d = env.render(offscreen=False, camera_name='gripperPOV', resolution=(img_size[0], img_size[1]),
                                mode='Depth_array')
        im12, im12_d = env.render(offscreen=False, camera_name='behindGripper', resolution=(img_size[0], img_size[1]),
                                mode='Depth_array')

    return im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12, im1_d, im2_d, im3_d, im4_d, im5_d, im6_d  # Le pasamos la im1 para que la guarde y haga GIFs en predict()

# Load model
model = torch.load(MODEL_NAME, map_location='cpu')

#####

# obs = env.reset()
obs = None
obs_prev = None
env_prev = None
done = False
i = 0
its = 0
finalize = 0
vector_init = []
vector_finalize = []
vector_finalize_bool = [0] * int(NUM_PRUEBAS)
vector_porcentaje_acierto = []
model_time = []
preprocessing_time = []

#scaler = joblib.load("MinMax_scaler.save")
num_iters_seguidas = 0

# Condicion de inicio
print('Reset Episode')

SEED = 10
env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['window-open-v2-goal-observable'](seed=SEED)
policy = SawyerWindowOpenV2Policy()
obs = env.reset()  # Reset environment
obs_data = None

action_prev = np.array([0, 0, 0, 0], dtype="float32")
obs_act = obs[0:4]
obs_prev = np.array([0, 0, 0, 0], dtype="float32")
obs_prev2 = np.array([0, 0, 0, 0], dtype="float32")

vector_init.append(i)

done = False
its = its + 1

nearness = 0 # Vamos a queres que una vez este cerca, se siga acercando diez iteraciones mas

while (not done) or (int(NUM_PRUEBAS) > int(its)):
    print("Reinicio cuando = 0 -> ", str(int(NUM_ITERS) - int(num_iters_seguidas)))
    print("Pruebas realizadas correctamente: ", int(finalize), "/", int(NUM_PRUEBAS))
    print(vector_finalize_bool)
    if ((int(NUM_ITERS) - int(num_iters_seguidas) == 0) or (done == True)):

        print('Reset Episode')
        print('Accion previa:')

        SEED = 10 + int(its)
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['window-open-v2-goal-observable'](seed=SEED)
        policy = SawyerWindowOpenV2Policy()
        obs = env.reset()  # Reset environment
        obs_data = None

        action_prev = np.array([0, 0, 0, 0], dtype="float32")
        obs_act = obs[0:4]
        obs_prev = np.array([0, 0, 0, 0], dtype="float32")
        obs_prev2 = np.array([0, 0, 0, 0], dtype="float32")
        print(action_prev)
        print(obs_act)
        print(obs_prev)
        print(obs_prev2)

        num_iters_seguidas = 0

        vector_init.append(i)

        if (done == True):
            vector_finalize_bool[int(its)-1] = 1
        done = False
        its = its + 1
        env_prev = env

    if len(action_prev) == 4:
        action_prev = action_prev[:-1]


    action = predict(model, env, env_prev, action_prev, obs_act, obs_prev, obs_prev2, i, SAVE)
    action_no_norm = np.array(action, dtype="float32")

    if NORM == str(1):  # En caso de tener que desnormalizar las acciones
        # print(action)
        action = np.array(action[:-1])
        action = action[np.newaxis, :]
        action = scaler.inverse_transform(action)
        # print(action)
        action = np.append(action, 1)

    env_prev = env

    '''
    ##### Nuevo
    print('Gripper: ', str(float(action_no_norm[-1])))
    if float(action_no_norm[-1]) <= 0.05:
        action_no_norm[-1] = 0
    else:
        action_no_norm[-1] = 0.1
    ##### Nuevo
    '''

    action_no_norm_aux = list(action_no_norm)
    action_no_norm_aux.append(0.1)
    action_no_norm = np.array(action_no_norm_aux, dtype='float32')

    obs, reward, done, info = env.step(action_no_norm)
    obs_prev2 = obs_prev
    obs_prev = obs_act
    obs_act = obs[0:4]
    near_object = float(info['success'])
    near_object2 = float(info['in_place_reward'])

    if near_object == 1.0 or near_object2 >= 0.20:
        nearness +=1

    if nearness >= 5:
        done = True
        nearness = 0

    action_prev = action_no_norm

    print('Acciones desde t hasta t-5:')
    print(action_prev)
    print(obs_act)
    print(obs_prev)
    print(obs_prev2)

    print('Nearness: ', str(nearness))
    print('Done: ', str(near_object))

    print(info)
    print(near_object,near_object2)

    i = i + 1
    num_iters_seguidas = num_iters_seguidas + 1

    if done == True:
        finalize = finalize + 1
        num_iters_seguidas = 0

    print('\nIteracion: ', i)
    print('Iteraciones seguidas: ', num_iters_seguidas)
    print("ITS: ", int(its))
    print("Num Pruebas: ", int(NUM_PRUEBAS))

    if its > int(NUM_PRUEBAS):
        done = True

    print('Terminate :', done)

vector_pruebas = []
header = ['Inicio', 'Final', 'Intentos', 'Finaliza', 'Acierto']

CSV_NAME = str(MODEL_NAME) + ".csv"
PATH = 'Resultados/' + str(CSV_NAME)

for i in range(len(vector_init)-1):
    vector_finalize.append(int(vector_init[i + 1]) - 1)
vector_finalize.append(i-1)

print(len(vector_init))
print(vector_init)
print(len(vector_finalize))
print(vector_finalize)
print(len(vector_pruebas))
print(vector_pruebas)
print(len(vector_finalize_bool))
print(vector_finalize_bool)
print(len(vector_porcentaje_acierto))
print(vector_porcentaje_acierto)

for i in range(int(NUM_PRUEBAS)):
    vector_pruebas.append(vector_finalize[i] - vector_init[i] + 1)

sum_cum = 0
for i in range(len(vector_init) - 1):
    sum_cum += vector_finalize_bool[i]
    x = sum_cum * 100
    value = str(round(float(x/(1+i)))) + "%"
    vector_porcentaje_acierto.append(value)

print(len(vector_init))
print(vector_init)
print(len(vector_finalize))
print(vector_finalize)
print(len(vector_pruebas))
print(vector_pruebas)
print(len(vector_finalize_bool))
print(vector_finalize_bool)
print(len(vector_porcentaje_acierto))
print(vector_porcentaje_acierto)

with open(CSV_NAME, 'w') as f:
    write = csv.writer(f)
    write.writerow(header)
    for i in range(len(vector_init) - 1):
        row = []
        row.append(vector_init[i])
        row.append(vector_finalize[i])
        row.append(vector_pruebas[i])
        row.append(vector_finalize_bool[i])
        row.append(vector_porcentaje_acierto[i])
        write.writerow(row)
'''


with open(CSV_NAME, 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(vector_finalize[:-1])
    write.writerow(vector_pruebas)
    write.writerow(vector_finalize_bool[:-1])
 '''
shutil.move(CSV_NAME, PATH)

print("Done!")

# Tiempo del Bloque 1: Preprocesamiento
min_pre = min(preprocessing_time)
max_pre = max(preprocessing_time)
mean_pre = sum(preprocessing_time) / len(preprocessing_time)
median_pre = median(preprocessing_time)

print("#####################################\nDatos del Bloque 1: Preprocesamiento\n#####################################")
plt.figure(figsize=(15,5))
plt.title("Histograma para el Bloque 1", fontdict ={'size':20})
plt.ylabel("Cantidad de iteraciones", fontdict ={'size':15})
plt.xlabel("Tiempo (milisegundos)", fontdict ={'size':15})
plt.hist(preprocessing_time, bins=200, range=(min(preprocessing_time),max(preprocessing_time)))
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.savefig(PATH + "Histograma_Bloque_1.png")

plt.figure(figsize=(15,5))
plt.title("Evolucion del tiempo por iteraciones para el Bloque 1", fontdict ={'size':20})
plt.ylabel("Tiempo (milisegundos)", fontdict ={'size':15})
plt.xlabel("Iteraciones", fontdict ={'size':15})
plt.plot(list(range(len(preprocessing_time))),preprocessing_time,"o", markersize=4, color="red")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.savefig(PATH + "Evolucion_Bloque_1.png")

print("Estadisticos iniciales:")
print(" - Minimo: " + str(min_pre) + " en la iteracion numero " + str(preprocessing_time.index(min_pre)))
print(" - Maximo: " + str(max_pre) + " en la iteracion numero " + str(preprocessing_time.index(max_pre)))
print(" - Media: " + str(mean_pre))
print(" - Mediana: " + str(median_pre))

# Tiempo del Bloque 2: Prediccion de los datos
mean_mod = sum(model_time) / len(model_time)
median_mod = median(model_time)
min_mod = min(model_time)
max_mod = max(model_time)

print("\n############################################\nDatos del Bloque 2: Prediccion de los datos\n############################################")
plt.figure(figsize=(15,5))
plt.title("Histograma para el Bloque 2", fontdict ={'size':20})
plt.ylabel("Cantidad de iteraciones", fontdict ={'size':15})
plt.xlabel("Tiempo (milisegundos)", fontdict ={'size':15})
plt.hist(model_time, bins=200, range=(min(model_time),max(model_time)))
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.savefig(PATH + "Histograma_Bloque_2.png")

plt.figure(figsize=(15,5))
plt.title("Evolucion del tiempo por iteraciones para el Bloque 2", fontdict ={'size':20})
plt.ylabel("Tiempo (milisegundos)", fontdict ={'size':15})
plt.xlabel("Iteraciones", fontdict ={'size':15})
plt.plot(list(range(len(model_time))),model_time,"o", markersize=4, color="red")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.savefig(PATH + "Evolucion_Bloque_2.png")

print("Estadisticos iniciales:")
print(" - Minimo: " + str(min_mod) + " en la iteracion numero " + str(model_time.index(min_mod)))
print(" - Maximo: " + str(max_mod) + " en la iteracion numero " + str(model_time.index(max_mod)))
print(" - Media: " + str(mean_mod))
print(" - Mediana: " + str(median_mod))

