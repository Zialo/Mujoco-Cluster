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
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

img_size = [128, 128]
NORM = sys.argv[1] if len(sys.argv) > 1 else 0
NUM_PRUEBAS = sys.argv[2] if len(sys.argv) > 2 else 10
NUM_ITERS = sys.argv[3] if len(sys.argv) > 3 else 100
SAVE = sys.argv[4] if len(sys.argv) > 4 else 1
OPTION = sys.argv[5] if len(sys.argv) > 5 else False
MODEL_NAME = sys.argv[6] if len(sys.argv) > 6 else 'model_pytorch_01593'

IMAGEN1 = ((MODEL_NAME.split("_")[2]).split(".")[0]).split("+")[0]
IMAGEN2 = ((MODEL_NAME.split("_")[2]).split(".")[0]).split("+")[1]
IMAGEN3 = ((MODEL_NAME.split("_")[2]).split(".")[0]).split("+")[2]
IMAGEN4 = ((MODEL_NAME.split("_")[2]).split(".")[0]).split("+")[3]
IMAGEN5 = ((MODEL_NAME.split("_")[2]).split(".")[0]).split("+")[4]
IMAGEN = str(IMAGEN1) + ' + ' + str(IMAGEN2) + ' + ' + str(IMAGEN3) + ' + ' + str(IMAGEN4) + ' + ' + str(IMAGEN5) + 'Window'
print(IMAGEN1)
print(IMAGEN2)
print(IMAGEN3)
print(IMAGEN4)
print(IMAGEN5)

if OPTION != True and OPTION != False:
    OPTION = True

print('Parametros introducidos: \n -NORM: ', str(NORM), '\n -NUM_PRUEBAS: ', str(NUM_PRUEBAS), '\n -NUM_ITERS: ', str(NUM_ITERS), '\n -OPTION: ', str(OPTION), '\n -SAVE: ', str(SAVE), '\n -MODEL: ', str(MODEL_NAME), '\n')

# Preparo carpeta de Train
if not os.path.exists("Imagenes GIF " + str(IMAGEN) ):
    os.makedirs("Imagenes GIF " + str(IMAGEN))
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Top")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Corner")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Corner2")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Corner3")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Gripper")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/BehindGripper")
else:
    shutil.rmtree("Imagenes GIF " + str(IMAGEN))
    os.makedirs("Imagenes GIF " + str(IMAGEN))
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Top")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Corner")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Corner2")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Corner3")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/Gripper")
    os.makedirs("Imagenes GIF " + str(IMAGEN) + "/BehindGripper")

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
        x1, x2 = x

        x2 = torch.from_numpy(x2)
        x2 = x2.unsqueeze(0)
        x2 = x2.to(torch.float32)

        f1 = self.fe(x1)
        f = self.flatten(self.avg_pool(f1))

        f = torch.cat((f, x2), dim=1)
        return self.clf(f)

def predict(model, env, env_prev, action_prev, i, SAVE):
    model = model.eval()
    # Get image from observation
    im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12 = get_image_from_observation(env, env_prev, i)

    if int(SAVE) == 1:
        # Save three first Images
        name_1 = "Imagenes GIF " + str(IMAGEN) + "/Top/" + str(i) + ".png"
        name_2 = "Imagenes GIF " + str(IMAGEN) + "/Corner/" + str(i) + ".png"
        name_3 = "Imagenes GIF " + str(IMAGEN) + "/Corner2/" + str(i) + ".png"
        name_4 = "Imagenes GIF " + str(IMAGEN) + "/Corner3/" + str(i) + ".png"
        name_5 = "Imagenes GIF " + str(IMAGEN) + "/Gripper/" + str(i) + ".png"
        name_6 = "Imagenes GIF " + str(IMAGEN) + "/BehindGripper/" + str(i) + ".png"

        top = Image.fromarray(im1)
        corner = Image.fromarray(im2)
        corner2 = Image.fromarray(im3)
        corner3 = Image.fromarray(im4)
        gripper = Image.fromarray(im5)
        behindgripper = Image.fromarray(im6)

        top.save(name_1, "PNG")
        corner.save(name_2, "PNG")
        corner2.save(name_3, "PNG")
        corner3.save(name_4, "PNG")
        gripper.save(name_5, "PNG")
        behindgripper.save(name_6, "PNG")

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49005188, 0.38129396, 0.34909529),
                             (0.17641072, 0.22982312, 0.23390036))
    ])
    
    if IMAGEN1 == 'Top':
        im_1 = im1
        im_6 = im7
    elif IMAGEN1 == 'Corner':
        im_1 = im2
        im_6 = im8
    elif IMAGEN1 == 'Corner2':
        im_1 = im3
        im_6 = im9
    elif IMAGEN1 == 'Corner3':
        im_1 = im4
        im_6 = im10
    elif IMAGEN1 == 'Gripper':
        im_1 = im5
        im_6 = im11
    elif IMAGEN1 == 'Behindgripper' or 'BehindGripper':
        im_1 = im6
        im_6 = im12
    
    if IMAGEN2 == 'Top':
        im_2 = im1
        im_7 = im7
    elif IMAGEN2 == 'Corner':
        im_2 = im2
        im_7 = im8
    elif IMAGEN2 == 'Corner2':
        im_2 = im3
        im_7 = im9
    elif IMAGEN2 == 'Corner3':
        im_2 = im4
        im_7 = im10
    elif IMAGEN2 == 'Gripper':
        im_2 = im5
        im_7 = im11
    elif IMAGEN2 == 'Behindgripper' or 'BehindGripper':
        im_2 = im6
        im_7 = im12
    
    if IMAGEN3 == 'Top':
        im_3 = im1
        im_8 = im7
    elif IMAGEN3 == 'Corner':
        im_3 = im2
        im_8 = im8
    elif IMAGEN3 == 'Corner2':
        im_3 = im3
        im_8 = im9
    elif IMAGEN3 == 'Corner3':
        im_3 = im4
        im_8 = im10
    elif IMAGEN3 == 'Gripper':
        im_3 = im5
        im_8 = im11
    elif IMAGEN3 == 'Behindgripper' or 'BehindGripper':
        im_3 = im6
        im_8 = im12
        
    if IMAGEN4 == 'Top':
        im_4 = im1
        im_9 = im7
    elif IMAGEN4 == 'Corner':
        im_4 = im2
        im_9 = im8
    elif IMAGEN4 == 'Corner2':
        im_4 = im3
        im_9 = im9
    elif IMAGEN4 == 'Corner3':
        im_4 = im4
        im_9 = im10
    elif IMAGEN4 == 'Gripper':
        im_4 = im5
        im_9 = im11
    elif IMAGEN4 == 'Behindgripper' or 'BehindGripper':
        im_4 = im6
        im_9 = im12
        
    if IMAGEN5 == 'Top':
        im_5 = im1
        im_10 = im7
    elif IMAGEN5 == 'Corner':
        im_5 = im2
        im_10 = im8
    elif IMAGEN5 == 'Corner2':
        im_5 = im3
        im_10 = im9
    elif IMAGEN5 == 'Corner3':
        im_5 = im4
        im_10 = im10
    elif IMAGEN5 == 'Gripper':
        im_5 = im5
        im_10 = im11
    elif IMAGEN5 == 'Behindgripper' or 'BehindGripper':
        im_5 = im6
        im_10 = im12

    im1 = transformations(im_1)
    im2 = transformations(im_2)
    im3 = transformations(im_3)
    im4 = transformations(im_4)
    im5 = transformations(im_5)
    im6 = transformations(im_6)
    im7 = transformations(im_7)
    im8 = transformations(im_8)
    im9 = transformations(im_9)
    im10 = transformations(im_10)
    
    im_actual = torch.cat((im1, im2, im3, im4, im5), axis=0)
    im_previa = torch.cat((im6, im7, im8, im9, im10), axis=0)
    image = torch.cat((im_actual, im_previa), axis=0)
    image = image.unsqueeze(dim=0)

    with torch.no_grad():
        pred = model((image, action_prev))
    pred = pred.detach().cpu().numpy()
    prediction = pred[0, :].tolist()
    #prediction.append(1)
    # print("PRED", prediction, " FIN")
    return prediction

def get_image_from_observation(env, env_prev, i):
    im1 = env.render(offscreen=True, camera_name='topview', resolution=(img_size[0], img_size[1]))
    im2 = env.render(offscreen=True, camera_name='corner', resolution=(img_size[0], img_size[1]))
    im3 = env.render(offscreen=True, camera_name='corner2', resolution=(img_size[0], img_size[1]))
    im4 = env.render(offscreen=True, camera_name='corner3', resolution=(img_size[0], img_size[1]))
    im5 = env.render(offscreen=True, camera_name='gripperPOV', resolution=(img_size[0], img_size[1]))
    im6 = env.render(offscreen=True, camera_name='behindGripper', resolution=(img_size[0], img_size[1]))

    if ((i % int(NUM_ITERS) == 0) or (i == 0)):
        im7 = im1
        im8 = im2
        im9 = im3
        im10 = im4
        im11 = im5
        im12 = im6

    else:
        im7 = env_prev.render(offscreen=True, camera_name='topview', resolution=(img_size[0], img_size[1]))
        im8 = env_prev.render(offscreen=True, camera_name='corner', resolution=(img_size[0], img_size[1]))
        im9 = env_prev.render(offscreen=True, camera_name='corner2', resolution=(img_size[0], img_size[1]))
        im10 = env_prev.render(offscreen=True, camera_name='corner3', resolution=(img_size[0], img_size[1]))
        im11 = env_prev.render(offscreen=True, camera_name='gripperPOV', resolution=(img_size[0], img_size[1]))
        im12 = env_prev.render(offscreen=True, camera_name='behindGripper', resolution=(img_size[0], img_size[1]))

    return im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12  # Le pasamos la im1 para que la guarde y haga GIFs en predict()

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

#scaler = joblib.load("MinMax_scaler.save")
num_iters_seguidas = 0

# Condicion de inicio
print('Reset Episode')

SEED = 10
env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['window-open-v2-goal-observable'](seed=SEED)
policy = SawyerWindowOpenV2Policy()
obs = env.reset()  # Reset environment
obs_data = None

action_prev = np.array([0, 0, 0], dtype="float32")

vector_init.append(i)

done = False
its = its + 1
obs_prev = obs
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

        action_prev = np.array([0, 0, 0], dtype="float32")
        print(action_prev)

        num_iters_seguidas = 0

        vector_init.append(i)

        if (done == True):
            vector_finalize_bool[int(its)-1] = 1
        done = False
        its = its + 1
        obs_prev = obs
        env_prev = env
    
    if len(action_prev) == 4:
        action_prev = action_prev[:-1]

    action = predict(model, env, env_prev, action_prev, i, SAVE)
    action_no_norm = np.array(action, dtype="float32")

    if NORM == str(1):  # En caso de tener que desnormalizar las acciones
        # print(action)
        action = np.array(action[:-1])
        action = action[np.newaxis, :]
        action = scaler.inverse_transform(action)
        # print(action)
        action = np.append(action, 1)

    obs_prev = obs
    env_prev = env

    '''
    ##### Nuevo
    print('Gripper: ', str(float(action_no_norm[-1])))
    if float(action_no_norm[-1]) <= 0.05:
        action_no_norm[-1] = 0
    else:
        action_no_norm[-1] = 0.1
    ##### Nuevo
    
    
    action_no_norm_aux = list(action_no_norm)
    action_no_norm_aux.append(0.1)
    action_no_norm = np.array(action_no_norm_aux, dtype='float32')
    '''
    action_no_norm_aux = list(action_no_norm)
    action_no_norm_aux.append(0.1)
    action_no_norm = np.array(action_no_norm_aux, dtype='float32')

    obs, reward, done, info = env.step(action_no_norm)
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

    print('Nearness: ', str(nearness))
    print('Done: ', str(near_object))
    
    #print(info)
    #print(near_object,near_object2)

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
#####
