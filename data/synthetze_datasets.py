from sklearn import datasets
import numpy as np
from custom_shapes import torus, circle_embedded
import pandas as pd

def main(object = 'circle'):
    if object == 'circle':
        sr_points, sr_color = circle_embedded(n=3000, ambient=3)
    elif object =='torus':
        sr_points = torus(n=3000, c=4, a=1)
        sr_color = []
    elif object == 'swiss_roll':
        sr_points, sr_color = datasets.make_swiss_roll(n_samples=3000, random_state=0)
    else:
        mammoth_all = pd.read_csv('mammoth_a.csv')
        sr_points = mammoth_all.sample(3000)
        sr_points = sr_points.to_numpy()
        sr_color = []

    data = np.zeros([len(sr_points[:, 0]),100])
    data_set={'data': [], 'label':[], 'color':[]}
    data_set['color'] = sr_color
    data_set['label'] = sr_points


    data[:,0] = sr_points[:,0] + sr_points[:,1] + sr_points[:, 2]
    data[:,1] = sr_points[:,0] + sr_points[:,1] - sr_points[:, 2]
    data[:,2] = sr_points[:,0] - sr_points[:,1] + sr_points[:, 2]
    data[:,3] = -sr_points[:,0] + sr_points[:,1] + sr_points[:, 2]

    for i in range(24):
        if object == 'circle':
            sr_points, _ = circle_embedded(n=3000, ambient=3)
        elif object == 'torus':
            sr_points = torus(n=3000, c=4, a=1)
        elif object == 'swiss_roll':
            sr_points, _ = datasets.make_swiss_roll(n_samples=3000)
        else:
            mammoth_all = pd.read_csv('mammoth_a.csv')
            sr_points = mammoth_all.sample(3000)
            sr_points = sr_points.to_numpy()
        data[:, 0 + 4*i + 4] = sr_points[:, 0] + sr_points[:, 1] + sr_points[:, 2]
        data[:, 1 + 4*i + 4] = sr_points[:, 0] + sr_points[:, 1] - sr_points[:, 2]
        data[:, 2 + 4*i + 4] = sr_points[:, 0] - sr_points[:, 1] + sr_points[:, 2]
        data[:, 3 + 4*i + 4] = -sr_points[:, 0] + sr_points[:, 1] + sr_points[:, 2]

    data_set['data'] = data
    np.save(object+'.npy', data_set)


if __name__ == "__main__":
    # object = 'circle'
    # object = 'torus'
    object = 'swiss_roll'
    # object = 'mammoth'

    main(object)