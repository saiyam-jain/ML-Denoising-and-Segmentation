import argparse
import numpy as np
import minterpy as mp
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from stardist import random_label_cmap
from stardist.matching import matching
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.BlackboxFunction import BlackboxFunction
from hyppopy.solvers.HyperoptSolver import HyperoptSolver

def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def loss_function(data, params, img_size, lbl_gt, model, img):
    global grid_index
    global hyper_param_keys
    global reg_tree
    global sdt
    coeffs = np.array([params[key] for key in hyper_param_keys])
    reg_tree.child[grid_index].regressor._regression_values = reg_tree.child[grid_index].regressor.regression_matrix@coeffs
    reg_vals = reg_tree.reconstruct_image(sdt)
    noise = reg_vals.reshape((img_size, img_size))
    rescaled_coeffs = (coeffs - noise.min())/(noise.max()-noise.min())*0.5 - 0.25
    reg_tree.child[grid_index].regressor._regression_values = reg_tree.child[grid_index].regressor.regression_matrix@rescaled_coeffs
    reg_vals = reg_tree.reconstruct_image(sdt)
    noise = reg_vals.reshape((img_size, img_size))
    new_img = (img - noise).copy()
    new_img[new_img<0] = 0
    new_lbl, _ = model.predict_instances(normalize(new_img))
    metrics =  matching(lbl_gt, new_lbl, thresh=thresh)
    return -metrics.f1

def main():
    i = np.load('number.npy')

    number = i-1
    np.save('number', number)

    model = StarDist2D.from_pretrained('2D_paper_dsb2018')

    data = np.load('data-cropped.npz')
    images = data['X']
    mask_gt = data['Y']

    lbl_cmap = random_label_cmap()

    total = len(images)
    print('total image count', total)
    print("Starting noise num: ", i)
    img = images[i]
    print('img.shape', img.shape)
    img = normalize(img)

    lbl_gt = mask_gt[i]
    noise_num = i

    mi = mp.MultiIndexSet.from_degree(spatial_dimension=2, poly_degree=poly_degree)

    num_coeffs = len(mi)
    rand_coeffs = np.random.random(num_coeffs)
    newt_poly = mp.NewtonPolynomial(mi, rand_coeffs)
    img_size = 256
    xvals = np.linspace(-1,1,img_size)
    yvals = np.linspace(-1,1,img_size)

    X,Y = np.meshgrid(xvals, yvals)

    coords = np.c_[X.reshape(-1),Y.reshape(-1)]

    function_values = newt_poly(coords).reshape(img_size,img_size)

    sdt = mp.DecompositionTree(
        block_shape=(img_size,img_size),
        spatial_dimension = 2,
        lp_degree=2.0,
        poly_degree=poly_degree
    )
    sdt.clear()
    sdt.subdivide(grid_size*grid_size, child_block_shape=(img_size//grid_size, img_size//grid_size), child_poly_degree = poly_degree)
    sdt.show()
    reg_tree = mp.windowed_regression_2D(function_values, sdt)
    reg_tree.show()

    lbl_pred, _ = model.predict_instances(img)
    metrics =  matching(lbl_gt, lbl_pred, thresh=thresh)
    best_f1 = metrics.f1
    best_noise = 0

    hyper_param_keys = []
    hyper_params = {}

    for j in range(num_coeffs):
        key = 'coeff_{:03}'.format(j)
        hyper_param_keys.append(key)
        hyper_params[key] = {
            "domain": "uniform",
            "data": [-coeffs_range, coeffs_range],
            "type": float
        }

    grid_index = 0
    grid_count = 0

    grid_num = np.arange(grid_size * grid_size)
    np.random.shuffle(grid_num)

    for j in 100, 200:
        num_iterations = j
        config = {
            "hyperparameter": hyper_params,
            "max_iterations": num_iterations,
            "solver": "hyperopt",
        }
        for ka in grid_num:
            grid_count+=1
            print('grid_count: ', grid_count)
            grid_index = ka
            project = HyppopyProject(config=config)
            solver = HyperoptSolver(project)
            blackbox = BlackboxFunction(data=[], blackbox_func=loss_function)
            solver.blackbox = blackbox
            solver.run()
            df, best = solver.get_results()
            best_coeffs = np.array(list(best.values()))
            reg_tree.child[grid_index].regressor._regression_values = reg_tree.child[grid_index].regressor.regression_matrix@best_coeffs
            reg_vals = reg_tree.reconstruct_image(sdt)
            noise = reg_vals.reshape((img_size, img_size))
            rescaled_coeffs = (best_coeffs - noise.min())/(noise.max()-noise.min())*0.5 - 0.25 #highlight
            reg_tree.child[grid_index].regressor._regression_values = reg_tree.child[grid_index].regressor.regression_matrix@rescaled_coeffs #highlight
            reg_vals = reg_tree.reconstruct_image(sdt)
            noise = reg_vals.reshape((img_size, img_size))
            new_img = (img - noise).copy()
            new_img[new_img < 0] = 0 #highlight set negative values to 0
            new_lbl, _ = model.predict_instances(normalize(new_img))
            metrics =  matching(lbl_gt, new_lbl, thresh=thresh)
            if metrics.f1>=best_f1:
                best_f1 = metrics.f1
                best_noise = noise
                path = f'noises/noise-{noise_num:05}'
                np.save(path, best_noise)
            if best_f1 == 1:
                break
        if best_f1 == 1:
            break

    new_img = (img - best_noise).copy()
    new_img[new_img<0] = 0
    new_lbl, _ = model.predict_instances(normalize(new_img))

    plt.figure(figsize=(15,11), constrained_layout=True)
    plt.subplot(231)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("input image")
    plt.subplot(232)
    plt.imshow(new_img, cmap="gray")
    plt.axis("off")
    plt.title("new input image (image - noise)")
    plt.subplot(233)
    plt.imshow(new_lbl, cmap=lbl_cmap)
    plt.axis('off')
    plt.title('segmentation mask after Hyppopy')
    plt.subplot(234)
    plt.imshow(lbl_gt, cmap=lbl_cmap)
    plt.axis("off")
    plt.title("GT segmentation mask")
    plt.subplot(235)
    plt.imshow(noise, cmap='gray')
    plt.axis("off")
    plt.title("noise")
    plt.subplot(236)
    plt.imshow(lbl_pred, cmap=lbl_cmap)
    plt.axis("off")
    plt.title("segmentation mask before Hyppopy")
    plt.savefig(f"plots/{noise_num:05}.png")

    path = f'masks/mask-{noise_num:05}'
    np.save(path, new_lbl)
    print('Finished for noise num: ', i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, help="iterations", default=50)
    parser.add_argument("--coeffs", type=int, help="range of coefficients", default=100)
    parser.add_argument("--poly_degree", type=int, help="polynomial degree", default=13)
    parser.add_argument("--grid_size", type=int, help="grid size", default=8)
    parser.add_argument("--thresh", type=float, help="thresh", default=0.75)
    args = parser.parse_args()

    num_iterations = args.iterations
    coeffs_range = args.coeffs
    poly_degree = args.poly_degree
    grid_size = args.grid_size
    thresh = args.thresh
    main()