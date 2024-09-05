import numpy as np
import pandas as pd
import sklearn.metrics as mtr
from stardist.matching import matching
from stardist.models import StarDist2D
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.BlackboxFunction import BlackboxFunction
from hyppopy.solvers.GridsearchSolver import GridsearchSolver

def separate_cells(mask):
    detected_cells = []
    
    for i in np.unique(mask).tolist():
        if i == 0:
            pass
        else:
            corr = np.where(mask == i)
            yy, xx = corr
            x1,y1, x2,y2 = xx.min(), yy.min(), xx.max(), yy.max()
            yc, xc = int(yy.mean()), int(xx.mean())
            # cell = gth[corr]
            cell_bound = {
                'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                'xc':xc,'yc':yc,
            }
            detected_cells.append(cell_bound)
    return detected_cells

def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def my_callback_function(**kwargs):
    print("\r{}".format(kwargs), end="")
    f = open("params3.txt","a")
    f.write( str(kwargs) )
    f.close()

def loss_function(data, params):
    distance = params['distance']
    percentage = params['percentage']
    ratio = params['ratio']
    f1_scores=[]
    for idx in range(13):
        img = normalize(valid_images[idx])
        gt = valid_masks[idx]
        star_dist = model.predict_instances(img)[0] # type: ignore
        detected_cells_seg = separate_cells(star_dist)
        df_seg = pd.DataFrame(detected_cells_seg)
        fp = np.zeros(star_dist.shape, dtype='int32') # type: ignore
        fn = np.zeros(star_dist.shape, dtype='int32') # type: ignore
        temp=1
        for noise in noises:
            new_img = (img - noise*percentage).copy()
            new_img[new_img<0] = 0
            defence = model.predict_instances(normalize(new_img))[0] # type: ignore
            detected_cells_dfs = separate_cells(defence)
            df_dfs = pd.DataFrame(detected_cells_dfs)   
            dis = mtr.pairwise_distances(df_seg, df_dfs)
            
            # Loop through each row
            for i, row in enumerate(dis):
                min_value = np.min(row)  # Find the minimum value in the row
                min_column_index = np.argmin(row)  # Find the index of the minimum value in the row
                # Check if the minimum value is also the minimum in its own column
                if not min_value == np.min(dis[:, min_column_index]) or np.min(dis[:, min_column_index])>distance:
                    fp[np.where(np.isin(star_dist, i+1))] = i+1

            # Loop through each col
            for i, col in enumerate(dis.T):
                min_value = np.min(col)  # Find the minimum value in the col
                min_row_index = np.argmin(col)  # Find the index of the minimum value in the col
                # Check if the minimum value is also the minimum in its own row
                if not min_value == np.min(dis[min_row_index, :]) or np.min(dis[min_row_index, :])>distance:
                    # Count the number of zeros in 'fn' for each corresponding position where 'defence' is 'i+1'
                    zeros_count = np.sum(np.logical_and(fn == 0, defence == i+1), axis=None)
                    
                    # Count the number of non-zeros in 'fn' for each corresponding position where 'defence' is 'i+1'
                    non_zeros_count = np.sum(np.logical_and(fn != 0, defence == i+1), axis=None)

                    if zeros_count>non_zeros_count*ratio:
                        fn[np.where(np.isin(defence, i+1))] = temp
                        temp+=1
                    else:
                        fn[np.where(np.isin(defence, i+1))] = fn[np.where(np.isin(defence, i+1))].max()
            
        fix_fp = star_dist-fp
        f1_fixed_fp = matching(gt, fix_fp, thresh=0.5).f1 # type: ignore

        fn[fn!=0] += (star_dist.max()+1) # type: ignore
        fix_fn = star_dist+fn
        f1_fixed_fn = matching(gt, fix_fn, thresh=0.5).f1 # type: ignore

        f1_scores.append(np.mean([f1_fixed_fp, f1_fixed_fn]))

    return -np.mean(f1_scores)

model = StarDist2D.from_pretrained('2D_paper_dsb2018')

test_images = np.load('test_images.npy')
test_masks = np.load('test_masks.npy')

valid_images=[]
valid_masks=[]

for i in range(len(test_images)):
    img = normalize(test_images[i])
    gt = test_masks[i]
    pred = model.predict_instances(img)[0] # type: ignore
    f1=matching(pred, gt, thresh=0.5).f1 # type: ignore
    if f1<0.95:
        valid_images.append(img)
        valid_masks.append(gt)

noises = []
for i in range(0, 300):
    try:
        noise = np.load(f'noises/noise-{i:05}.npy')
    except Exception as error:
        print(type(error).__name__, "–", error)
        continue

    noises.append(noise)

# Create the HyppopyProject class instance
project = HyppopyProject()
project.add_hyperparameter(name="distance", domain="uniform", data=[3, 20], frequency=18, type=int)
project.add_hyperparameter(name="percentage", domain="uniform", data=[0.1, 1], frequency=10, type=float)
project.add_hyperparameter(name="ratio", domain="normal", data=[1, 3], frequency=15, type=float)

solver = GridsearchSolver(project=project)
blackbox = BlackboxFunction(data=[], blackbox_func=loss_function, callback_func=my_callback_function)
solver.blackbox = blackbox
solver.run()
df, best = solver.get_results()

print("\n")
print("*"*100)
print(df)
print("\n")
print("*"*100)
print("Best Parameter Set:\n{}".format(best))
print("*"*100)