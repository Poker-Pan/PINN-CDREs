from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import Deepxde.mywork.draw_picture as Draw
import Deepxde.deepxde as dde
from Deepxde.deepxde.backend import tf
import Fem_ML.Fem.Solver_Steady_ConvectionDiffusionEquation_3D as FEM_Steady_Solver


def main():

    def pde(x, y):
        X, Y, Z = x[:, 0:1], x[:, 1:2], x[:, 2:]
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_y, dy_z = dy_x[:, 0:1], dy_x[:, 1:2], dy_x[:, 2:]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
        dy_yy = tf.gradients(dy_y, x)[0][:, 1:2]
        dy_zz = tf.gradients(dy_z, x)[0][:, 2:]
        beta_1, beta_2, beta_3 = 0.0, 3, -Y
        epsilon = 1e-3
        gamma = 1

        return -epsilon * (dy_xx + dy_yy + dy_zz) + (beta_1 * dy_x + beta_2 * dy_y + beta_3 * dy_z) + gamma * y

    def boundary_0(x, on_boundary):
        return on_boundary & \
               ((x[2] == 0.5) |
                ((x[1] <= 0.5) & (x[2] == 0.3)) | ((x[2] == 0) & (x[1] >= 0.5)) |
                ((x[1] == 0.5) & (x[2] <= 0.3))
                |
                ((x[0] == 1) & ((x[1] >= 0.5) | ((x[1] <= 0.5) & (x[2] >= 0.3)))) |
                ((x[0] == 0) & ((x[1] >= 0.5) | ((x[1] <= 0.5) & (x[2] >= 0.3)))))

    def boundary_1(x, on_boundary):  # left
        return on_boundary & (x[1] == 0) & (x[2] >= 0.3)

    def boundary_2(x, on_boundary):  # right
        return on_boundary & (x[1] == 2)

    def value_0(x):
        num_data = x.shape[0]
        return np.zeros((num_data, 1))

    kappa = 0.1
    def value_1(x):
        num_data = x.shape[0]
        return kappa*(0.5 - (x[:, 2]) * (x[:, 2] - 0.3)).reshape(num_data, 1)

    def value_2(x):
        num_data = x.shape[0]
        return np.zeros((num_data, 1))


    geom1 = dde.geometry.Cuboid(xmin=[0, 0, 0.3], xmax=[1, 0.5, 0.5])
    geom2 = dde.geometry.Cuboid(xmin=[0, 0.5, 0], xmax=[1, 2, 0.5])
    geom = dde.geometry.CSGUnion(geom1, geom2)
    bc_0 = dde.DirichletBC(geom, value_0, boundary_0)
    bc_1 = dde.DirichletBC(geom, value_1, boundary_1)
    bc_2 = dde.NeumannBC(geom, value_2, boundary_2)  #NeumannBC

    percent = 50
    eta = 1e-5
    N_hidden = 4
    N_neuron = 100
    data = dde.data.PDE(geom, pde, [bc_0, bc_1, bc_2], num_domain=int(100*percent),
                        num_boundary=int(20*percent), num_test=5000, wight=[1, 1, 10])

    net = dde.maps.FNN([3] + [N_neuron] * N_hidden + [1], "tanh", "Glorot uniform")     ## generate model  "elu" "relu" "selu" "sigmoid" "sin" "swish" "tanh"
    model = dde.Model(data, net)

    model.compile("adam", lr=eta)
    losshistory, train_state = model.train(epochs=20000)
    # model.compile("L-BFGS-B")
    # losshistory, train_state = model.train()
    # dde.saveplot(losshistory, train_state, issave=False, isplot=True)

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
    Numerical = 1
    if Numerical == 1:
        ## Numerical 1
        h1, h2, h3 = 0.1, 0.5, 0.9
        y1 = np.linspace(0, 2, 100, endpoint=True)
        y2 = np.linspace(0.5, 2, 75, endpoint=True)
        z1 = np.linspace(0.3, 0.5, 25, endpoint=True)
        z2 = np.linspace(0, 0.3, 25, endpoint=True)

        # x1, x2 = h1 * np.ones((25, 100)), h1 * np.ones((25, 75))
        x11, x22 = h2 * np.ones((25, 100)), h2 * np.ones((25, 75))
        # x111, x222 = h3 * np.ones((25, 100)), h3 * np.ones((25, 75))
        y1, z1 = np.meshgrid(y1, z1)
        y2, z2 = np.meshgrid(y2, z2)
        # coordinates1 = np.hstack((x1.reshape((-1, 1)), y1.reshape((-1, 1)), z1.reshape((-1, 1))))
        # coordinates2 = np.hstack((x2.reshape((-1, 1)), y2.reshape((-1, 1)), z2.reshape((-1, 1))))
        coordinates11 = np.hstack((x11.reshape((-1, 1)), y1.reshape((-1, 1)), z1.reshape((-1, 1))))
        coordinates22 = np.hstack((x22.reshape((-1, 1)), y2.reshape((-1, 1)), z2.reshape((-1, 1))))
        # coordinates111 = np.hstack((x111.reshape((-1, 1)), y1.reshape((-1, 1)), z1.reshape((-1, 1))))
        # coordinates222 = np.hstack((x222.reshape((-1, 1)), y2.reshape((-1, 1)), z2.reshape((-1, 1))))

        # ml_z1 = model.predict(coordinates1).reshape(25, 100)
        # ml_z2 = model.predict(coordinates2).reshape(25, 75)
        ml_z11 = model.predict(coordinates11).reshape(25, 100)
        ml_z22 = model.predict(coordinates22).reshape(25, 75)
        # ml_z111 = model.predict(coordinates111).reshape(25, 100)
        # ml_z222 = model.predict(coordinates222).reshape(25, 75)

        ## draw picture
        mins = min(ml_z11.min(), ml_z22.min())
        maxs = max((ml_z11 - mins).max(), (ml_z22 - mins).max())
        # mins = min(ml_z1.min(), ml_z2.min(), ml_z11.min(), ml_z22.min(), ml_z111.min(), ml_z222.min())
        # maxs = max((ml_z1 - mins).max(), (ml_z2 - mins).max(), (ml_z11 - mins).max(),
        #            (ml_z22 - mins).max(), (ml_z111 - mins).max(), (ml_z222 - mins).max())
        # color_ml1 = (ml_z1 - mins) / maxs
        # color_ml2 = (ml_z2 - mins) / maxs
        color_ml11 = (ml_z11 - mins) / maxs
        color_ml22 = (ml_z22 - mins) / maxs
        # color_ml111 = (ml_z111 - mins) / maxs
        # color_ml222 = (ml_z222 - mins) / maxs

        ## -----------------------------------------------------------------------------------------------------------------
        fig = plt.figure() #figsize=(4.5, 3.5)
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.35)

        ## ------------------------------
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=25, azim=25)
        ax1.set_xlabel('x'), ax1.set_ylabel('y'), ax1.set_zlabel('z')
        ax1.set_xlim((0, 1)), ax1.set_ylim((0, 2)), ax1.set_zlim((0, 0.5))
        ax1.set_box_aspect([0.5, 1, 0.25])
        ax1.zaxis.set_major_locator(MultipleLocator(0.1))
        # ax1.plot_surface(x1, y1, z1, facecolors=plt.cm.jet(color_ml1), linewidth=0, antialiased=False, shade=False)
        # ax1.plot_surface(x2, y2, z2, facecolors=plt.cm.jet(color_ml2), linewidth=0, antialiased=False, shade=False)
        ax1.plot_surface(x11, y1, z1, facecolors=plt.cm.jet(color_ml11), linewidth=0, antialiased=False, shade=False)
        ax1.plot_surface(x22, y2, z2, facecolors=plt.cm.jet(color_ml22), linewidth=0, antialiased=False, shade=False)
        # ax1.plot_surface(x111, y1, z1, facecolors=plt.cm.jet(color_ml111), linewidth=0, antialiased=False, shade=False)
        # ax1.plot_surface(x222, y2, z2, facecolors=plt.cm.jet(color_ml222), linewidth=0, antialiased=False, shade=False)

        M = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        # M.set_array(np.hstack((x1, x2, x11, x22, x111, x222)))
        M.set_array(np.hstack((x11, x22)))
        position = fig.add_axes([0.15, 0.15, 0.7, 0.03])
        cb1 = fig.colorbar(M, cax=position, orientation='horizontal')
        cb1.set_ticks(np.linspace(M.norm.vmin, M.norm.vmax, num=5))
        # scale = np.linspace(min(ml_z1.min(), ml_z2.min(), ml_z11.min(), ml_z22.min(), ml_z111.min(), ml_z222.min()),
        #                     max(ml_z1.max(), ml_z2.max(), ml_z11.max(), ml_z22.max(), ml_z111.max(), ml_z222.max()), num=10)
        scale = np.linspace(min(ml_z11.min(), ml_z22.min()),
                            max(ml_z11.max(), ml_z22.max()), num=5)
        scale1 = []
        for i in range(0, 5):
            scale1.append('{:.2e}'.format(scale[i]))
        cb1.set_ticklabels(scale1)

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'
        plt.savefig('Num4_kappa=' + str(kappa) + '.pdf')
        plt.show()
        # os.system('shutdown -s -f -t 60')
        w = 1


    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(data.train_x[:, 0], data.train_x[:, 1], data.train_x[:, 2])
    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
    # plt.show()



if __name__ == "__main__":
    main()
