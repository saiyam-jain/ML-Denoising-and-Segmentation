import cv2
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from stardist.models import StarDist2D

def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def get_gaussian_defence(img, model, scale=1.0):
    shape = (256, 256)
    min_value = -0.25
    max_value = 0.25
    mean = 0.0
    std_dev = 0.01

    gaussian_noise = np.random.normal(mean, std_dev, shape)

    gaussian_noise_scaled = gaussian_noise * ((max_value - min_value) / (gaussian_noise.max() - gaussian_noise.min()))
    gaussian_noise_scaled -= gaussian_noise_scaled.mean()
    gaussian_noise_scaled += (max_value + min_value) / 2

    gaussian_noise_scaled = np.clip(gaussian_noise_scaled, min_value, max_value)

    new_img = (img - gaussian_noise_scaled*scale).copy()
    new_img[new_img<0] = 0
    gaussian_defence = model.predict_instances(normalize(new_img))[0]

    return(gaussian_defence)

def get_poisson_defence(img, model, scale=1.0):
    shape = (256, 256)  # Shape of the Poisson noise pattern
    lam = 0.1  # Lambda parameter of the Poisson distribution
    min_value = -0.25  # Minimum value of the range
    max_value = 0.25  # Maximum value of the range

    # Generate Poisson noise
    poisson_noise = np.random.poisson(lam, shape)

    # Scale the values to fit within the range [-0.25, 0.25]
    poisson_noise_scaled = (poisson_noise - np.min(poisson_noise)) / (np.max(poisson_noise) - np.min(poisson_noise)) * (max_value - min_value) + min_value
    poisson_noise_scaled = np.clip(poisson_noise_scaled, min_value, max_value)

    new_img = (img - poisson_noise_scaled*scale).copy()
    new_img[new_img<0] = 0
    poisson_defence = model.predict_instances(normalize(new_img))[0]

    return(poisson_defence)

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

def draw_cell_bbox(img, cells):
    bbox_img = img.copy()
    for cell_bound in cells:
        cv2.rectangle(bbox_img, (cell_bound['x1'], cell_bound['y1']), (cell_bound['x2'], cell_bound['y2']), int(img.max()), 1)
    return bbox_img

def find_same_point(df1, df2, threshold):
    dis = mtr.pairwise_distances(df1, df2)
    corr = np.where(dis < threshold)
    matched1 = df1.loc[corr[0], :]
    matched2 = df2.loc[corr[1], :]
    non_matched1 = df1.loc[df1.index.difference(corr[0]), :]
    non_matched2 = df2.loc[df2.index.difference(corr[1]), :]
    return corr, matched1, matched2, non_matched1, non_matched2

def plot_figures(figures, titles, main_title=None, nrows=None, ncols=None, figsize=(15, 15)):
    num_figures = len(figures)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Flatten the axes if there's only one row or column
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_figures):
        ax = axes[i // ncols, i % ncols]
        ax.imshow(figures[i])
        ax.set_title(titles[i])
        # ax.axis('off')
    
    if main_title is not None:
        fig.suptitle(main_title, fontsize=12)
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'path.png')

def get_fp_fn(img_number, percentage):
    img = normalize(valid_images[img_number])
    gt = valid_masks[img_number]
    pred = model.predict_instances(img)[0]
    detected_cells_seg = separate_cells(pred)
    df_seg = pd.DataFrame(detected_cells_seg)
    fn = np.zeros(pred.shape, dtype='int32')
    fp = np.zeros(pred.shape, dtype='int32')
    temp=1
    for noise in noises:
        new_img = (img - noise*percentage).copy()
        new_img[new_img<0] = 0
        defence = model.predict_instances(normalize(new_img))[0]
        detected_cells_dfs = separate_cells(defence)
        df_dfs = pd.DataFrame(detected_cells_dfs)   
        dis = mtr.pairwise_distances(df_seg, df_dfs)

        # Loop through each row
        for i, row in enumerate(dis):
            min_value = np.min(row)  # Find the minimum value in the row
            min_column_index = np.argmin(row)  # Find the index of the minimum value in the row
            if not min_value == np.min(dis[:, min_column_index]) or np.min(dis[:, min_column_index])>14:
                fp[np.where(np.isin(pred, i+1))] = i+1

        # Loop through each col
        for i, col in enumerate(dis.T):
            min_value = np.min(col)  # Find the minimum value in the col
            min_row_index = np.argmin(col)  # Find the index of the minimum value in the col
            if not min_value == np.min(dis[min_row_index, :]) or np.min(dis[min_row_index, :])>14:
                # Count the number of zeros in 'fn' for each corresponding position where 'defence' is 'i+1'
                zeros_count = np.sum(np.logical_and(fn == 0, defence == i+1), axis=None)
                
                # Count the number of non-zeros in 'fn' for each corresponding position where 'defence' is 'i+1'
                non_zeros_count = np.sum(np.logical_and(fn != 0, defence == i+1), axis=None)

                if zeros_count>non_zeros_count*1.7:
                    fn[np.where(np.isin(defence, i+1))] = temp
                    temp+=1
                else:
                    fn[np.where(np.isin(defence, i+1))] = fn[np.where(np.isin(defence, i+1))].max()

    return fp, fn

def get_fp_fn_gauss_poisson(img_number):
    img = normalize(valid_images[img_number])
    gt = valid_masks[img_number]
    pred = model.predict_instances(img)[0]
    detected_cells_seg = separate_cells(pred)
    df_seg = pd.DataFrame(detected_cells_seg)
    fn_gauss = np.zeros(pred.shape, dtype='int32')
    fp_gauss = np.zeros(pred.shape, dtype='int32')
    temp=1
    for idx in range(250):
        gauss_defence = get_gaussian_defence(img, model, scale=0.4)
        detected_cells_dfs = separate_cells(gauss_defence)
        df_dfs = pd.DataFrame(detected_cells_dfs)   
        dis = mtr.pairwise_distances(df_seg, df_dfs)
        for i, row in enumerate(dis):
            min_value = np.min(row)  # Find the minimum value in the row
            min_column_index = np.argmin(row)  # Find the index of the minimum value in the row
            if not min_value == np.min(dis[:, min_column_index]) or np.min(dis[:, min_column_index])>12:
                fp_gauss[np.where(np.isin(pred, i+1))] = i+1
        for i, col in enumerate(dis.T):
            min_value = np.min(col)  # Find the minimum value in the col
            min_row_index = np.argmin(col)  # Find the index of the minimum value in the col
            if not min_value == np.min(dis[min_row_index, :]) or np.min(dis[min_row_index, :])>12:
                zeros_count = np.sum(np.logical_and(fn_gauss == 0, gauss_defence == i+1), axis=None)
                non_zeros_count = np.sum(np.logical_and(fn_gauss != 0, gauss_defence == i+1), axis=None)
                if zeros_count>non_zeros_count*1.7:
                    fn_gauss[np.where(np.isin(gauss_defence, i+1))] = temp
                    temp+=1
                else:
                    fn_gauss[np.where(np.isin(gauss_defence, i+1))] = fn_gauss[np.where(np.isin(gauss_defence, i+1))].max()

    fn_poisson = np.zeros(pred.shape, dtype='int32')
    fp_poisson = np.zeros(pred.shape, dtype='int32')
    temp=1
    for idx in range(250):
        poisson_defence = get_poisson_defence(img, model, scale=0.4)
        detected_cells_dfs = separate_cells(poisson_defence)
        df_dfs = pd.DataFrame(detected_cells_dfs)   
        dis = mtr.pairwise_distances(df_seg, df_dfs)
        for i, row in enumerate(dis):
            min_value = np.min(row)  # Find the minimum value in the row
            min_column_index = np.argmin(row)  # Find the index of the minimum value in the row
            if not min_value == np.min(dis[:, min_column_index]) or np.min(dis[:, min_column_index])>12:
                fp_poisson[np.where(np.isin(pred, i+1))] = i+1
        for i, col in enumerate(dis.T):
            min_value = np.min(col)  # Find the minimum value in the col
            min_row_index = np.argmin(col)  # Find the index of the minimum value in the col
            if not min_value == np.min(dis[min_row_index, :]) or np.min(dis[min_row_index, :])>12:
                zeros_count = np.sum(np.logical_and(fn_poisson == 0, poisson_defence == i+1), axis=None)
                non_zeros_count = np.sum(np.logical_and(fn_poisson != 0, poisson_defence == i+1), axis=None)
                if zeros_count>non_zeros_count*1.7:
                    fn_poisson[np.where(np.isin(poisson_defence, i+1))] = temp
                    temp+=1
                else:
                    fn_poisson[np.where(np.isin(poisson_defence, i+1))] = fn_poisson[np.where(np.isin(poisson_defence, i+1))].max()

    return fp_gauss, fn_gauss, fp_poisson, fn_poisson

if __name__ == "__main__":
    model = StarDist2D.from_pretrained('2D_paper_dsb2018')

    test_images = np.load('test_images.npy')
    test_masks = np.load('test_masks.npy')

    noises = []
    for i in range(0, 300):
        try:
            noise = np.load(f'noises/noise-{i:05}.npy')
        except Exception as error:
            print(type(error).__name__, "–", error)
            continue

        noises.append(noise)

    valid_images=[]
    valid_masks=[]
    f1_scores=[]

    for i in range(len(test_images)):
        img = normalize(test_images[i])
        gt = test_masks[i]
        pred = model.predict_instances(img)[0] # type: ignore
        f1=matching(pred, gt, thresh=0.5).f1 # type: ignore
        if f1<0.95:
            print(i, f1)
            f1_scores.append(f1)
            valid_images.append(img)
            valid_masks.append(gt)

    fp, fn = get_fp_fn(0, 0.1)
    fp_gauss, fn_gauss, fp_poisson, fn_poisson = get_fp_fn_gauss_poisson(0)
    img = normalize(valid_images[0])
    gt = valid_masks[0]
    pred = model.predict_instances(img)[0]

    bb1 = draw_cell_bbox(img, separate_cells(fp))
    bb2 = draw_cell_bbox(img, separate_cells(fn))
    bb3 = draw_cell_bbox(img, separate_cells(fp_gauss))
    bb4 = draw_cell_bbox(img, separate_cells(fn_gauss))
    bb5 = draw_cell_bbox(img, separate_cells(fp_poisson))
    bb6 = draw_cell_bbox(img, separate_cells(fn_poisson))
    comparisonGP = np.array([img, gt, pred, bb1, bb2, gt, bb3, bb4, bb5, bb6])

    plot_figures(comparisonGP,
                titles=['Raw', 'gt', 'StarDist', 'poly_def_FP', 'poly_def_FN', 'gt', 'gauss_def_fp', 'gauss_def_fn', 'poisson_def_fp', 'poisson_def_fn'], 
                nrows=2, ncols=5, figsize=(14,8))