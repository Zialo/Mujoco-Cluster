import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
import os
from PIL import Image
plt.style.use('dark_background')
import pandas as pd
import fnmatch

class MyAnimation:
    def __init__(self, data):
        self.fig = plt.figure()
        self.data = np.array(data)
        self.ax = plt.axes(xlim=(0, self.data.shape[0]), ylim=(self.data.min()-0.1, self.data.max()+0.1))
        self.lines = [self.ax.plot([], [], lw=2, label=f'pos{i}') for i in range(self.data.shape[1])]
        self.lines = list(map(lambda x: x[0], self.lines))
        plt.legend()
        self.xdata = [[] for i in range(self.data.shape[1])]
        self.ydata = [[] for i in range(self.data.shape[1])]
        
    def init_animation(self):
        # creating an empty plot/frame
        for i in range(len(self.lines)):
            self.lines[i].set_data([], [])
        return self.lines
    
    def animate(self, i): 
        # appending new points to x, y axes points list 
        for j in range(self.data.shape[1]):
            self.xdata[j].append(i)
            self.ydata[j].append(self.data[i, j])
            self.lines[j].set_data(self.xdata[j], self.ydata[j])
        return self.lines

# Actions
'''
sub_df = df.iloc[:49, :]
a = MyAnimation(sub_df)
anim = animation.FuncAnimation(a.fig, a.animate, init_func=a.init_animation, frames=sub_df.shape[0], interval=20, blit=True) 
# save the animation as mp4 video file 
anim.save('rlbench_actions.gif',writer='imagemagick') 
'''
aux=list(np.arange(len(os.listdir("Imagenes GIF ALL Button/Corner"))))
aux2=[]
for i in aux:
    aux2.append(str(aux[i])+".png")
    
inicio = [73,282,343,766,1000]
final = [134,340,421,839,1084]
# Images
for p in range(len(inicio)):

    fig = plt.figure()
    name = "Imagenes GIF ALL Button/Gif_" + str(p) + ".gif"
    all_imagesCorner = [Image.open(os.path.join("Imagenes GIF ALL Button/Corner/", i)) for i in aux2[inicio[p]:final[p]]]
    all_imagesCorner2 = [Image.open(os.path.join("Imagenes GIF ALL Button/Corner2/", i)) for i in aux2[inicio[p]:final[p]]]
    all_imagesTop = [Image.open(os.path.join("Imagenes GIF ALL Button/Top/", i)) for i in aux2[inicio[p]:final[p]]]

    #inicio = final + 1
    #final = final + 100

    tam=all_imagesCorner[0].size
    im=Image.new("RGB",(tam[0]*3,tam[1]))

    plt_images = []
    for i in range(len(all_imagesCorner)):
        im.paste(all_imagesCorner[i],(0,0))
        im.paste(all_imagesCorner2[i],(tam[0],0))
        im.paste(all_imagesTop[i],(tam[0]*2,0))

        im2=plt.imshow(im, animated=True)

        plt_images.append([im2])

    for j in range(1):
        plt_images.append([im2])
    aniConcatenada = animation.ArtistAnimation(fig, plt_images, interval=60, blit=True)
    aniConcatenada.save(name, writer='imagemagick')
    print(name + " DONE")
