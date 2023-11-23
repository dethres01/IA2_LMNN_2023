import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import numpy as np
import threading

# Multi layer neural network

class MLN:

    def __init__(self,eta=0.1, epoch=1000, min_error=0.01, n_neurons=6):
        self.lr = eta
        self.epoch = epoch
        self.n_epoch = 0
        self.min_error = min_error
        self.n_neurons = n_neurons
        # init weights
        self.w_hidden = np.matrix(np.random.rand(self.n_neurons, 3))
        self.w_output = np.random.rand(self.n_neurons+1)
        # X is the points, d is the desired output
        # keep track of errors
        self.X = []
        self.d = []
        self.errors = []
        # gui stuff
        self.fig_e, self.ax_e = plt.subplots()
        self.fig, self.ax = plt.subplots()
        self.canvas = None
        self.canvas_e = None

    def set_canvas(self):
        mainwindow = Toplevel()
        mainwindow.wm_title("MLN")
        mainwindow.geometry("990x600")
        #add image to window
        img = PhotoImage(file="bg.png")
        img_label = Label(mainwindow, image=img, bg='white')
        img_label.place(x=0, y=0, relwidth=1, relheight=1)

        # add canvas

        self.canvas = FigureCanvasTkAgg(self.fig, master=mainwindow)
        self.canvas.get_tk_widget().place(x=520, y=120, width=480, height=480)
        self.fig.canvas.mpl_connect('button_press_event', self.set_dots)

        # error canvas
        self.canvas_e = FigureCanvasTkAgg(self.fig_e, master=mainwindow)
        self.canvas_e.get_tk_widget().place(x=20, y=70, width=300, height=200)
        execute_button = Button(mainwindow, text="Train Function",command=lambda: threading.Thread(target=self.train).start())
        execute_button.place(x=740, y=50)
        jacobian_button = Button(mainwindow, text="Train Jacobian",command=lambda: threading.Thread(target=self.train_jacobian).start())
        jacobian_button.place(x=740, y=80)
        title = Label(mainwindow, text="MLN", font=("Helvetica", 16))
        title.place(x=80, y=10)
        self.set_axis()
        self.label_n_epoch = Label(mainwindow, text="Epoch: 0")
        self.label_n_epoch.place(x=10, y=300)
        self.label_error = Label(mainwindow, text="Error: 0")
        self.label_error.place(x=90, y=300)

        mainwindow.mainloop()


    def set_dots(self, event):
        ix, iy = event.xdata, event.ydata
        self.X.append((1, ix, iy))
        if event.button == 1:
            self.d.append(1)
            self.ax.plot(ix, iy, '.r')
        elif event.button == 3:
            self.d.append(0)
            self.ax.plot(ix, iy, '.b')
        self.canvas.draw()
    

    def activation(self, x, w):
        a = 1
        v = np.dot(x, w)
        f = 1/(1+np.exp(-a*v))
        return f
    
    def fd_activation(self, Y):
        a = 1
        f = a*Y*(1-Y)
        return f
    
    def calc_hidden(self):
        #print("X.shape", self.X.shape, "w_hidden.shape", np.transpose(self.w_hidden).shape)
        hidden_y = self.activation(self.X, np.transpose(self.w_hidden))
        # wx+b <- 
        hidden_x = np.c_[np.ones(len(hidden_y)), hidden_y]
        y = self.activation(hidden_x, np.array(self.w_output).flatten())
        return hidden_y, hidden_x, y
    
    def train(self):
        #we are going to try to calculate the jacobian matrix while training
        num_samples = len(self.X)
        jacobian_size = self.w_hidden.size + self.w_output.size
        print("jacobian size", jacobian_size)
        jacobian = np.zeros((num_samples, jacobian_size))
        # going for a hard stop on epoch or error
        error = True
        self.X = np.matrix(self.X)
        # weights already initialized
        while self.epoch and error:
            error = False
            #forward propagation
            hidden_y, hidden_x, y = self.calc_hidden()
            # calculate error
            errors = np.array(self.d) - y
            # calculate delta
            delta_output = []
            #back propagation
            for i in range(len(hidden_x)):
                dy = self.fd_activation(np.array(y).flatten()[i])
                delta_output.append(dy*np.array(errors).flatten()[i]) #this is the gradient

                self.w_output = self.w_output + np.dot(hidden_x[i], self.lr*delta_output[-1])
            for i in range(len(self.X)):
                for j in range(len(self.w_hidden)):
                    dy = self.fd_activation(hidden_y[i, j])
                    #derivada para delta
                    hidden_delta = np.array(self.w_output).flatten()[j+1]*np.array(delta_output).flatten()[i]*dy
                    #usamos delta siguiente porque es hidden
                    #actualizamos valores
                    self.w_hidden[j] = self.w_hidden[j] + np.dot(self.X[i], self.lr*hidden_delta)
            # check error

            square_error = np.average(np.power(errors, 2))

            if square_error > self.min_error:
                error = True
            self.errors.append(square_error)

            
            hidden_y, hidden_x, y = self.calc_hidden()
            self.plotting(y)
            self.epoch -= 1
            self.n_epoch += 1

    def Loss(self, y, d):
        return np.sum(np.power(d-y, 2))
    def fd_Loss(self, y, d):
        return 2*(d-y)
    def train_jacobian(self):
        #we are going to try to calculate the jacobian matrix while training
        num_samples = len(self.X)
        jacobian_size = self.w_hidden.size + self.w_output.size
        #print("jacobian size", jacobian_size)
        jacobian = np.zeros((num_samples, jacobian_size))
        # going for a hard stop on epoch or error
        error = True
        self.X = np.matrix(self.X)
        # weights already initialized
        lambda_factor = 0.1
        while self.epoch and error:
            error = False
            #forward propagation
            hidden_y, hidden_x, y = self.calc_hidden()
            # calculate error
            errors = np.array(self.d) - y
            #print("errors", errors.shape)
            print("errors", errors)

            for n in range(num_samples):
                #print("y[n]", np.array(y).flatten()[n])
                #print("d[n]", self.d[n])
                # jacobian matrix filling
                #dL_dy = -2 * (self.d[n]-np.array(y).flatten()[n]) # this is dL/dy which is the derivative of the loss function with respect to output
                dL_dy = -1
                dy_dv = self.fd_activation(np.array(y).flatten()[n]) # this is dy/dv which is the derivative of the activation function with respect to the output
                dv_dw = hidden_x[n] # this is dv/dw which is the derivative of the output with respect to the weights
                #print("dl_dy", dL_dy.shape)
                #print("dy_dv", dy_dv.shape)
                #print("dv_dw", dv_dw.shape)
                dL_dw = dL_dy*dy_dv*dv_dw# this is the derivative of the loss function with respect to the weights
                #print("dl_dw", dL_dw.shape)
                # jacobian matrix
                jacobian[n, :self.w_output.size] = dL_dw.flatten()
                #print("Hidden_y", hidden_y[n].flatten().T)
                #hidden layer 
                # we need dL_dy, dy_dv

                dv_dh = np.transpose(self.w_output) # this is dv/dh which is the derivative of the output with respect to the hidden layer
                # for each hidden neuron
                #print(self.n_neurons)
                for i in range(self.n_neurons):

                    #print("hidden_y", hidden_y[n,i])
                    #print("index", n,i)

                    dh_dvh = self.fd_activation(hidden_y[n,i])
                    #print("dv_dh", dv_dh.shape)
                    #print("dh_dvh", dh_dvh.shape)
                    dvh_dw = self.X[n]
                    #print("dvh_dw", dvh_dw.shape)
                    dL_dw_hidden = dL_dy*dy_dv*dv_dh[i]*dh_dvh*dvh_dw
                    #print("dL_dw_hidden", dL_dw_hidden.shape)
                    jacobian[n, self.w_output.size+i*self.w_hidden.shape[1]:self.w_output.size+(i+1)*self.w_hidden.shape[1]] = dL_dw_hidden.flatten()

                #dh_dvh = self.fd_activation(np.array(hidden_y).flatten()[n]) # this is dh/dvh which is the derivative of the activation function with respect to the hidden layer
                #dvh_dw = self.X[n] # this is dvh/dw which is the derivative of the hidden layer with respect to the weights

                #print("dvh_dw", dvh_dw.shape)
                

                #print("dL_dw_hidden", dL_dw_hidden.shape)
                #print("jacobian", jacobian.shape)
                #print("jacobian", jacobian)
            #update weights
            # Weights update = (J^T* lambda Factor*Identity)^-1*J^T*error
            print("jacobian", jacobian)

            jacobian_T = np.transpose(jacobian)

            identity = np.identity(jacobian_T.shape[0])
            #print("identity", identity.shape)

            gradient_descent = lambda_factor*identity
            #print("gradient_descent", gradient_descent.shape)

            hessian = np.dot(jacobian_T, jacobian) + gradient_descent

            #print("hessian", hessian.shape)

            hessian_inv = np.linalg.inv(hessian)

            #print("hessian_inv", hessian_inv.shape)

            delta =  np.dot(np.dot(hessian_inv, jacobian_T), errors.T)
            delta_1d = np.array(delta).flatten()
            #print("delta1d", delta_1d.shape)

            # save a copy of weights that isn't updated by changes in self.w_output and self.w_hidden
            temp_w_output = self.w_output.copy()
            temp_w_hidden = self.w_hidden.copy()
            #print("delta", delta[self.w_output.size:].shape)
            #print("w_hidden", self.w_hidden)
            square_error = np.average(np.power(errors, 2))
            #print("self.w_output", self.w_output.shape)
            #print("delta", delta[:self.w_output.size].shape)
            #print("self.w_output", self.w_output.shape)
            #reshape delta to a n_neurons/input_size matrix
            # split the rest of delta into n_neurons matrices of input_size
            # TIME TO USE A COUNTER BECAUSE WE ARE DESPERATE

            counter = 0
            #output layer
            for i in range(self.w_output.size):
            #    print("delta value", delta_1d[i])
            #    print("self.w_output[i]", self.w_output[i])
                self.w_output[i] = self.w_output[i] - delta_1d[i]
            #    print("self.w_output[i] after", self.w_output[i])
                counter += 1
        

            
            #print("self.w_output", self.w_output)
            #temp
            #print("temporary w_output", temp_w_output)

            #print("temporary w_hidden", temp_w_hidden)
            
            #print("counter", counter)
            for i in range(self.n_neurons):
            #    print("self.w_hidden[i]", self.w_hidden[i])
                for j in range(self.X.shape[1]):
            #        print("w_hidden[i,j]", self.w_hidden[i,j])
                    self.w_hidden[i,j] = self.w_hidden[i,j] - delta_1d[counter]
            #        print("w_hidden[i,j] after", self.w_hidden[i,j])
                    counter += 1
            #print("self.w_hidden", self.w_hidden)
            

            # check new error
            hidden_y, hidden_x, y_new = self.calc_hidden()
            errors = np.array(self.d) - y_new
            print("errors", errors)
            square_error_new = np.average(np.power(errors, 2))
            
            print("square_error", square_error)
            print("square_error_new", square_error_new)
            if square_error_new > square_error:
                square_error = square_error_new
                lambda_factor *= 1.1
            else:
                print("Is lower")
                lambda_factor *= 0.9

            
            #print("square_error", square_error)
            print("lambda_factor", lambda_factor)

            if square_error > self.min_error:
                error = True
            else:
                error = False
            self.errors.append(square_error)
            
            self.plotting_jacob()
            self.n_epoch += 1
            self.epoch -= 1

    def plotting_jacob(self):
        self.ax.cla()
        # predict
        _, _, y = self.calc_hidden()

        for i in range(len(np.array(y).flatten())):
            if np.array(y).flatten()[i] >= 0.5:
                self.ax.plot(self.X[i,1], self.X[i,2], '.r')
            else:
                self.ax.plot(self.X[i,1], self.X[i,2], '.b')

        x_v = np.linspace(-4, 4, 20)
        y_v = np.linspace(-4, 4, 20)
        # meshgrid
        meshX, meshY = np.meshgrid(x_v, y_v)
        meshZ = []
        for i in range(len(meshX)):
            xc = np.transpose([meshX[i], meshY[i]])
            xc = np.c_[np.ones(len(xc)), xc]
            hidden_y = self.activation(xc, np.transpose(self.w_hidden))
            hidden_x = np.c_[np.ones(len(hidden_y)), hidden_y]
            yc = self.activation(hidden_x, np.array(self.w_output).flatten())
            meshZ.append(np.array(yc).flatten())

        mapping = plt.get_cmap('viridis')
        self.ax.contourf(meshX, meshY, meshZ, cmap=mapping)
        self.ax_e.cla()
        self.ax_e.plot(self.errors, c='r')
        self.ax_e.set_xticklabels([])

        self.set_axis()

        self.canvas.draw()
        self.canvas_e.draw()
        self.label_n_epoch.config(text="Epoch: "+str(self.n_epoch))
        self.label_error.config(text="Error: "+str(self.errors[-1]))


    def set_axis(self):

        aX = [-4,4]
        aY = [-4,4]
        self.ax.grid('on')
        self.ax_e.grid('on')
        zeros = np.zeros(2)
        self.ax.plot(aX, zeros, c='k')
        self.ax.plot(zeros, aY, c='k')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)

    
    def plotting(self,y):
        self.ax.cla()

        for i in range(len(np.array(y).flatten())):

            if np.array(y).flatten()[i] >= 0.5:
                self.ax.plot(self.X[i,1], self.X[i,2], '.r')
            else:
                self.ax.plot(self.X[i,1], self.X[i,2], '.b')

        x_v = np.linspace(-4, 4, 20)
        y_v = np.linspace(-4, 4, 20)
        # meshgrid
        meshX, meshY = np.meshgrid(x_v, y_v)
        meshZ = []
        for i in range(len(meshX)):
            xc = np.transpose([meshX[i], meshY[i]])
            xc = np.c_[np.ones(len(xc)), xc]
            hidden_y = self.activation(xc, np.transpose(self.w_hidden))
            hidden_x = np.c_[np.ones(len(hidden_y)), hidden_y]
            yc = self.activation(hidden_x, np.array(self.w_output).flatten())
            meshZ.append(np.array(yc).flatten())
        mapping = plt.get_cmap('viridis')
        self.ax.contourf(meshX, meshY, meshZ, cmap=mapping)
        self.ax_e.cla()
        self.ax_e.plot(self.errors, c='r')
        self.ax_e.set_xticklabels([])

        self.set_axis()

        self.canvas.draw()
        self.canvas_e.draw()
        self.label_n_epoch.config(text="Epoch: "+str(self.n_epoch))
        self.label_error.config(text="Error: "+str(self.errors[-1]))
        


if __name__ == "__main__":
    #mln = MLN()
    #mln.set_canvas()
    mln = MLN()
    mln.set_canvas()
    #mln.train_jacobian()
