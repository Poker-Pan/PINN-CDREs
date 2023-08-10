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
    epsilon = 1e-12
    def pde(x, y):
        X, Y, Z = x[:, 0:1], x[:, 1:2], x[:, 2:]
        dy_x = tf.gradients(y, x)[0]
        dy_x, dy_y, dy_z = dy_x[:, 0:1], dy_x[:, 1:2], dy_x[:, 2:]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
        dy_yy = tf.gradients(dy_y, x)[0][:, 1:2]
        dy_zz = tf.gradients(dy_z, x)[0][:, 2:]
        beta_1, beta_2, beta_3 = 1.0-Z, 1.0-Z, 1.0
        # beta_1, beta_2, beta_3 = 1.0, 1.0, 1.0
        # beta_1, beta_2, beta_3 = 0.0, 0.0, 0.5
        gamma = 1

        return -epsilon * (dy_xx + dy_yy + dy_zz) + (tf.multiply(beta_1, dy_x) + tf.multiply(beta_2, dy_y) + tf.multiply(beta_3, dy_z))  #numerical 2
        # return -epsilon*(dy_xx + dy_yy + dy_zz) + (beta_1*dy_x + beta_2*dy_y + beta_3*dy_z) - ((-2*epsilon*(Y*(Y-1)*Z*(Z-1) + X*(X-1)*Z*(Z-1) + Y*(Y-1)*X*(X-1))) + (beta_1*(2*X-1)*Y*(Y-1)*Z*(Z-1) + beta_2*(2*Y-1)*X*(X-1)*Z*(Z-1) + beta_3*(2*Z-1)*Y*(Y-1)*X*(X-1)))   #nmerical 1
        # return -epsilon*(dy_xx + dy_yy + dy_zz) + (beta_1*dy_x + beta_2*dy_y + beta_3*dy_z) + gamma*y - \
        #     (2/(np.sqrt(epsilon)*np.pi)*X*Y*Z*tf.pow(1/(1+Z*Z/epsilon),2) + (beta_1*(Y/np.pi*tf.atan(Z/np.sqrt(epsilon))) + beta_2*(X/np.pi*tf.atan(Z/np.sqrt(epsilon))) + beta_3*(X*Y/(np.sqrt(epsilon)*np.pi*(1+Z*Z/epsilon)))) + gamma*(X * Y / (np.pi) * tf.atan(Z / np.sqrt(epsilon))) )
        # return -epsilon*(dy_xx + dy_yy + dy_zz) + (beta_1*dy_x + beta_2*dy_y + beta_3*dy_z) + gamma*y - \
        #         (-epsilon*tf.exp(X)*tf.exp(Y)*tf.exp(Z) + (beta_1*tf.exp(X)*tf.exp(Y)*tf.exp(Z) + beta_2*tf.exp(X)*tf.exp(Y)*tf.exp(Z) + beta_3*tf.exp(X)*tf.exp(Y)*tf.exp(Z)) + gamma*tf.exp(X)*tf.exp(Y)*tf.exp(Z))


    def boundary(_, on_boundary):
        return on_boundary

    def func(x):   # boundary condition
        X, Y, Z = x[:, 0:1], x[:, 1:2], x[:, 2:]
        value = 0*X
        for i in range(0, x.shape[0]):
            if X[i] <= 0.5 and Y[i] <= 0.5 and Z[i] == 0:
                value[i] = 1
            else:
                value[i] = 0
        # epsilon = 1e-3
        # value = X * Y / (np.pi) * np.arctan(Z / np.sqrt(epsilon))
        # value = np.exp(X) * np.exp(Y) * np.exp(Z)
        return value


    geom = dde.geometry.Cuboid(xmin=[0, 0, 0], xmax=[1, 1, 1])
    bc = dde.DirichletBC(geom, func, boundary)    # dde.NeumannBC   dde.OperatorBC    dde.PeriodicBC   dde.RobinBC

    percent = 6
    eta = 7e-6
    N_hidden = 4
    N_neuron = 50
    data = dde.data.PDE(geom, pde, bc, num_domain=int(1200*percent),
                        num_boundary=int(120*percent), num_test=8000, wight=[1, 1, 10])
    net = dde.maps.FNN([3] + [N_neuron] * N_hidden + [1], "tanh", "Glorot uniform")     ## generate model  "elu" "relu" "selu" "sigmoid" "sin" "swish" "tanh"
    model = dde.Model(data, net)

    model.compile("adam", lr=eta)
    losshistory, train_state = model.train(epochs=1)
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
        def true_fun(X, Y, Z):
            epsilon = 1e-3
            # return X * Y / (np.pi) * np.arctan(Z / np.sqrt(epsilon))
            return np.exp(X) * np.exp(Y) * np.exp(Z)

        ## Numerical 1
        h = 0.5
        Num = 50
        x = np.linspace(0, 1, Num, endpoint=True)
        y = np.linspace(0, 1, Num, endpoint=True)
        z = h * np.ones((Num, Num))
        x, y = np.meshgrid(x, y)
        coordinates = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))))


        true_value = []
        for i in range(0, x.reshape((-1, 1)).shape[0]):
            true_value.append(true_fun(x.reshape((-1, 1))[i, 0], y.reshape((-1, 1))[i, 0], z.reshape((-1, 1))[i, 0]))
        tv_z = np.array(true_value).reshape(Num, Num)
        # fem_z_all = FEM_Steady_Solver.FEM_Solver_CDE(1, 1, 2, 'predict', coordinates.T, data.test_x.T, np.array([[0, 0, 0], [0, 0, 0]]).T)[0:2]
        # fem_z_all_1 = np.array(fem_z_all[0])
        # fem_z_all_2 = np.array(fem_z_all[1])
        # np.save("Fem_0.05_1.npy", fem_z_all_1)  # 保存文件
        # np.save("Fem_0.05_2.npy", fem_z_all_2)  # 保存文件
        fem_z_all_1 = np.load("Fem_20_1.npy")  # 读取文件
        fem_z_all_2 = np.load("Fem_20_2.npy")  # 读取文件
        fem_z = np.array(fem_z_all_1).reshape(Num, Num)
        ml_z = model.predict(coordinates).reshape(Num, Num)

        ## draw picture
        rsfem_z = abs(tv_z - fem_z)
        rsml_z = abs(tv_z - ml_z)

        mins = min(tv_z.min(), ml_z.min(), fem_z.min())
        maxs = max((tv_z - mins).max(), (ml_z - mins).max(), (fem_z - mins).max())

        color_tv = (tv_z - mins) / maxs
        color_fem = (fem_z - mins) / maxs
        color_ml = (ml_z - mins) / maxs

        mins = min(rsfem_z.min(), rsml_z.min())
        maxs = max((rsfem_z - mins).max(), (rsml_z - mins).max())
        color_rsfem = (rsfem_z - mins) / maxs
        color_rsml = (rsml_z - mins) / maxs

        ## ------------------------------------------------------------------
        fig = plt.figure(figsize=(14, 7.0))
        # plt.suptitle('3D-ConvectionDiffusion Equation')
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.35)

        ## ------------------------------
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.view_init(elev=30, azim=30)
        ax1.set_title("$u_{True}$")
        ax1.set_xlim((0, 1)), ax1.set_ylim((0, 1)), ax1.set_zlim((0, 1))
        ax1.zaxis.set_major_locator(MultipleLocator(0.1))
        ax1.plot_surface(x, y, z, facecolors=plt.cm.jet(color_tv), linewidth=0, antialiased=False, shade=False)
        ## ------------------------------
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.view_init(elev=30, azim=30)
        ax2.set_title("$u_{DLM_{3d}}$")
        ax2.set_xlim((0, 1)), ax2.set_ylim((0, 1)), ax2.set_zlim((0, 1))
        ax2.zaxis.set_major_locator(MultipleLocator(0.1))
        ax2.plot_surface(x, y, z, facecolors=plt.cm.jet(color_ml), linewidth=0, antialiased=False, shade=False)
        ## ------------------------------
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.view_init(elev=30, azim=30)
        ax3.set_title("$u_{FEM}$")
        ax3.set_xlim((0, 1)), ax3.set_ylim((0, 1)), ax3.set_zlim((0, 1))
        ax3.zaxis.set_major_locator(MultipleLocator(0.1))
        ax3.plot_surface(x, y, z, facecolors=plt.cm.jet(color_fem), linewidth=0, antialiased=False, shade=False)

        ## ------------------------------
        ax4 = fig.add_subplot(234)
        loss_train = np.sum(np.array(losshistory.loss_train) * losshistory.loss_weights, axis=1)
        loss_test = np.sum(np.array(losshistory.loss_test) * losshistory.loss_weights, axis=1)
        p41 = ax4.semilogy(losshistory.steps, loss_train, label="Train loss")
        p42 = ax4.semilogy(losshistory.steps, loss_test, label="Test loss")
        ax4.yaxis.tick_right()
        plt.legend()
        ax4.set_xlabel("Steps")
        ## ------------------------------
        ax5 = fig.add_subplot(235, projection='3d')
        ax5.view_init(elev=30, azim=30)
        ax5.set_title("$|u_{True}-u_{DLM_{3d}}|$")
        ax5.set_xlim((0, 1)), ax5.set_ylim((0, 1)), ax5.set_zlim((0, 1))
        ax5.zaxis.set_major_locator(MultipleLocator(0.1))
        ax5.plot_surface(x, y, z, facecolors=plt.cm.Reds(color_rsml), linewidth=0, antialiased=False, shade=False)
        ## ------------------------------
        ax6 = fig.add_subplot(236, projection='3d')
        ax6.view_init(elev=30, azim=30)
        ax6.set_title("$|u_{True}-u_{FEM}|$")
        ax6.set_xlim((0, 1)), ax6.set_ylim((0, 1)), ax6.set_zlim((0, 1))
        ax6.zaxis.set_major_locator(MultipleLocator(0.1))
        ax6.plot_surface(x, y, z, facecolors=plt.cm.Reds(color_rsfem), linewidth=0, antialiased=False, shade=False)

        M, N = plt.cm.ScalarMappable(cmap=plt.cm.jet), plt.cm.ScalarMappable(cmap=plt.cm.Reds)
        M.set_array(z), N.set_array(z)
        cb1 = fig.colorbar(M, ax=[ax1, ax2, ax3])
        cb2 = fig.colorbar(N, ax=[ax4, ax5, ax6])

        cb1.set_ticks(np.linspace(M.norm.vmin, M.norm.vmax, num=10))
        scale = np.linspace(min(tv_z.min(), ml_z.min(), fem_z.min()), max(tv_z.max(), ml_z.max(), fem_z.max()), num=10)
        scale1 = []
        for i in range(0, 10):
            scale1.append('{:.2e}'.format(scale[i]))
        cb1.set_ticklabels(scale1)

        cb2.set_ticks(np.linspace(N.norm.vmin, N.norm.vmax, num=10))
        scale = np.linspace(min(rsfem_z.min(), rsml_z.min()), max(rsfem_z.max(), rsml_z.max()), num=10)
        scale2 = []
        for i in range(0, 10):
            scale2.append('{:.2e}'.format(scale[i]))
        cb2.set_ticklabels(scale2)

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'
        plt.savefig('Num1_' + str(percent) + '.pdf')
        plt.show()

        ## predict
        error_true = []
        for i in range(0, data.test_x.shape[0]):
            error_true .append(true_fun(data.test_x[i, 0], data.test_x[i, 1], data.test_x[i, 2]))
        error_true = np.array(error_true).reshape(-1)
        error_fem = np.array(fem_z_all_2).reshape(-1)
        error_ml = np.array(model.predict(data.test_x)).reshape(-1)
        L_2_true = np.linalg.norm(error_true, ord=2)
        L_2_ml = np.linalg.norm(error_true-error_ml, ord=2) / L_2_true
        L_2_fem = np.linalg.norm(error_true-error_fem, ord=2) / L_2_true
        print('ml:', L_2_ml)
        print('fem:', L_2_fem)
        # os.system('shutdown -s -f -t 60')

    elif Numerical == 2:
        ## Numerical 2
        def grad(x, y):
            X, Y, Z = x[:, 0:1], x[:, 1:2], x[:, 2:]
            dy_x = tf.gradients(y, x)[0]
            dy_x, dy_y, dy_z = dy_x[:, 0:1], dy_x[:, 1:2], dy_x[:, 2:]
            return dy_x, dy_y, dy_z
        h = 0.5
        Num = 50
        x = np.linspace(0, 1, Num, endpoint=True)
        z = np.linspace(0, 1, Num, endpoint=True)
        # y = h * np.ones((Num, Num))
        x, z = np.meshgrid(x, z)
        y = x
        coordinates = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1))))

        ml_z = model.predict(coordinates).reshape(Num, Num)
        # u_dx, u_dy, u_dz = model.predict(coordinates, operator=grad)
        # u_dx, u_dy, u_dz = u_dx.reshape(Num, Num), u_dy.reshape(Num, Num), u_dz.reshape(Num, Num)
        # d = [u_dx, u_dy, u_dz]

        Draw.Numerical_2(x, y, z, ml_z, losshistory, epsilon)

if __name__ == "__main__":
    main()
