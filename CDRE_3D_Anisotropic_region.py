from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import Deepxde.mywork.draw_picture as Draw
import Deepxde.deepxde as dde
from Deepxde.deepxde.backend import tf
import Fem_ML.Fem.Solver_Steady_ConvectionDiffusionEquation_3D as FEM_Steady_Solver


def main():
    beta_1, beta_2, beta_3 = 0.0, 10.0, 0.0
    epsilon = 1e-6
    def pde(x, y):
        X, Y, Z = x[:, 0:1], x[:, 1:2], x[:, 2:]
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        dy_zz = dde.grad.hessian(y, x, i=2, j=2)
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_y = dde.grad.jacobian(y, x, i=0, j=1)
        dy_z = dde.grad.jacobian(y, x, i=0, j=2)
        gamma = 1

        return -epsilon * (dy_xx + dy_yy + dy_zz) + (beta_1 * dy_x + beta_2 * dy_y + beta_3 * dy_z) + gamma * y

    def boundary_0(x, on_boundary):  # top
        return on_boundary & ((x[2] == 0.1) | (x[2] == 0) | (x[0] == 0) | (x[0] == 1))

    def boundary_1(x, on_boundary):  # left
        return on_boundary & (x[1] == 0)

    def boundary_2(x, on_boundary):  # right
        return on_boundary & (x[1] == 2)

    def value_0(x):
        num_data = x.shape[0]
        return np.zeros((num_data, 1))

    def value_1(x):
        num_data = x.shape[0]
        return (0.1 - (x[:, 2]) * (x[:, 2])).reshape(num_data, 1)


    geom = dde.geometry.Cuboid(xmin=[0, 0, 0], xmax=[1, 2, 0.1])
    bc_0 = dde.DirichletBC(geom, value_0, boundary_0)
    bc_1 = dde.DirichletBC(geom, value_1, boundary_1)
    bc_2 = dde.NeumannBC(geom, value_0, boundary_2)

    percent = 50
    eta = 1e-5
    N_hidden = 5
    N_neuron = 50
    data = dde.data.PDE(geom, pde, [bc_0, bc_1, bc_2], num_domain=int(100*percent),  #bc_01, bc_02, bc_03, bc_04
                        num_boundary=int(10*percent), num_test=8000, wight=[1, 10, 0])
    net = dde.maps.FNN([3] + [N_neuron] * N_hidden + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=eta)
    losshistory, train_state = model.train(epochs=20000)

    ## Calculate the number of parameters
    def Sum_parameters_number():
        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total number of trainable parameters: %d" % total_parameters)
    # Sum_parameters_number()

    ## -----------------------------------------------------------------------------------------------------------------
    ## predict and draw
    Numerical = 3
    if Numerical == 3:
        ## Numerical 1
        h = 0.5
        Num = 50
        z = np.linspace(0, 0.1, Num, endpoint=True)
        y = np.linspace(0, 2, 2*Num, endpoint=True)
        x1 = h * np.ones((Num, 2*Num))
        y1, z1 = np.meshgrid(y, z)
        coordinates = np.hstack((x1.reshape((-1, 1)), y1.reshape((-1, 1)), z1.reshape((-1, 1))))


        ml_z11 = model.predict(coordinates).reshape(Num, 2*Num)

        ## draw picture
        mins = ml_z11.min()
        maxs = (ml_z11 - mins).max()
        color_ml11 = (ml_z11 - mins) / maxs

        ## -----------------------------------------------------------------------------------------------------------------
        fig = plt.figure() #figsize=(4.5, 3.5)
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.35)

        ## ------------------------------
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=25, azim=15)
        ax1.set_xlabel('x'), ax1.set_ylabel('y'), ax1.set_zlabel('z')
        ax1.set_xlim((0, 1)), ax1.set_ylim((0, 2)), ax1.set_zlim((0, 0.1))
        ax1.set_box_aspect([0.5, 1, 0.05])
        ax1.zaxis.set_major_locator(MultipleLocator(0.1))
        ax1.plot_surface(x1, y1, z1, facecolors=plt.cm.jet(color_ml11), linewidth=0, antialiased=False, shade=False)

        M = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        M.set_array(x1)
        # position = fig.add_axes([0.82, 0.25, 0.02, 0.5])
        position = fig.add_axes([0.15, 0.25, 0.7, 0.03])
        cb1 = fig.colorbar(M, cax=position, orientation='horizontal')
        cb1.set_ticks(np.linspace(M.norm.vmin, M.norm.vmax, num=5))
        scale = np.linspace(ml_z11.min(), ml_z11.max(), num=5)
        scale1 = []
        for i in range(0, 5):
            scale1.append('{:.2e}'.format(scale[i]))
        cb1.set_ticklabels(scale1)


        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'
        plt.savefig('Num4_' + str(beta_2) + '.pdf')
        plt.show()
        # os.system('shutdown -s -f -t 60')
        w = 1
if __name__ == "__main__":
    main()
