import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import numpy as np
import threading

# Multi layer neural network

class MLN:

    def __init__(self,eta=0.1, epoch=1500, min_error=0.01, n_neurons=6):
        
        self.lr = eta #aprendizaje rate (learning rate)
        self.epoch = epoch #numero de epocas (iteraciones)
        self.n_epoch = 0 #numero de epocas (iteraciones) actuales (para gui)
        self.min_error = min_error #error minimo para parar el entrenamiento
        self.n_neurons = n_neurons #numero de neuronas en la capa oculta (hidden layer)
        # init weights
        self.w_hidden = np.matrix(np.random.rand(self.n_neurons, 3)) #pesos de la capa oculta (hidden layer)
        self.w_output = np.random.rand(self.n_neurons+1) #pesos de la capa de salida (output layer) +1 por el bias (sesgo)
        # X is the points, d is the desired output
        # keep track of errors
        self.X = [] #puntos de entrada (input) (x,y)
        self.d = [] #salida deseada (desired output) (0,1) (1,0) (0,0) (1,1) 
        self.errors = [] #errores de cada epoca (iteracion) (para gui) 
        # gui stuff
        self.fig_e, self.ax_e = plt.subplots() #figura y ejes para el error 
        self.fig, self.ax = plt.subplots() #figura y ejes para el grafico
        self.canvas = None #canvas para el grafico
        self.canvas_e = None #canvas para el error

    def set_canvas(self):
        # set window
        mainwindow = Toplevel()
        mainwindow.wm_title("Levenberg-Marquardt")
        mainwindow.geometry("990x600")
        mainwindow = Toplevel() #ventana principal
        mainwindow.wm_title("MLN") #titulo de la ventana
        mainwindow.geometry("1080x720") #tamano de la ventana
        #add image to window
        img = PhotoImage(file="bg.png") #imagen de fondo
        img_label = Label(mainwindow, image=img, bg='white') #label para la imagen de fondo
        img_label.place(x=0, y=0, relwidth=1, relheight=1) #posicion de la imagen de fondo 

        # add canvas

        self.canvas = FigureCanvasTkAgg(self.fig, master=mainwindow)
        self.canvas.get_tk_widget().place(x=520, y=120, width=580, height=580) #posicion del grafico en la ventana principal 
        self.fig.canvas.mpl_connect('button_press_event', self.set_dots) #evento para agregar puntos al grafico con el mouse

        # error canvas
        
        self.canvas_e = FigureCanvasTkAgg(self.fig_e, master=mainwindow) #canvas para el error
        self.canvas_e.get_tk_widget().place(x=20, y=70, width=300, height=200)
        execute_button = Button(mainwindow, text="Train Function",command=lambda: threading.Thread(target=self.train).start())
        execute_button.place(x=740, y=50)
        jacobian_button = Button(mainwindow, text="Train Jacobian",command=lambda: threading.Thread(target=self.train_jacobian).start())
        jacobian_button.place(x=740, y=80)
        title = Label(mainwindow, text="Levenberg-Marquardt", font=("Helvetica", 16))
        title.place(x=400, y=10)
        # reset button
        reset_button = Button(mainwindow, text="Reset", command=lambda: self.clear_data())
        reset_button.place(x=840, y=80)
        title = Label(mainwindow, text="MLN", font=("Helvetica", 16))
        title.place(x=80, y=10)
        self.set_axis()
        self.label_n_epoch = Label(mainwindow, text="Epoch: 0")
        self.label_n_epoch.place(x=10, y=300)
        self.label_error = Label(mainwindow, text="Error: 0")
        self.label_error.place(x=10, y=320)

        mainwindow.mainloop()
    

    def clear_data(self):
        # Clear existing data
        self.X = []
        self.d = []
        self.errors = []
        self.n_epoch = 0
        self.epoch = 1500
        self.n_neurons = 6
        self.w_hidden = np.matrix(np.random.rand(self.n_neurons, 3))
        self.w_output = np.random.rand(self.n_neurons+1)
        self.label_n_epoch.config(text="Epoch: 0")
        self.label_error.config(text="Error: 0")
        self.ax.cla()
        self.ax_e.cla()
        self.set_axis()
        self.canvas.draw()
        self.canvas_e.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.set_dots)
        self.fig_e.canvas.mpl_connect('button_press_event', self.set_dots)
        self.canvas.draw()
        self.canvas_e.draw()
        #self.label_n_epoch.config(text="Epoch: "+str(self.n_epoch))
        #self.label_error.config(text="Error: "+str(self.errors[-1]))
        
    # set dots on canvas and add to X and d arrays for training later
    def set_dots(self, event):
        
        ix, iy = event.xdata, event.ydata #posicion del mouse en el grafico 
        self.X.append((1, ix, iy)) #agregamos el punto a la lista de puntos de entrada (input)
        if event.button == 1: #si el boton izquierdo del mouse es presionado 
            self.d.append(1) #agregamos el punto a la lista de salidas deseadas (desired output)
            self.ax.plot(ix, iy, marker='o', color='red') #graficamos el punto en el grafico
            #self.ax.plot(ix, iy, '.r')
        elif event.button == 3:
            self.d.append(0) #agregamos el punto a la lista de salidas deseadas (desired output)
            self.ax.plot(ix, iy, marker='x', color='blue')
            #self.ax.plot(ix, iy, '.b')
        self.canvas.draw()
    

    def activation(self, x, w):
        
        a = 1
        v = np.dot(x, w)
        f = 1/(1+np.exp(-a*v))
        return f
    
    def fd_activation(self, Y):
        
        a = 1 #factor de activacion
        f = a*Y*(1-Y) #derivada de la funcion de activacion (sigmoid)
        return f
    
    def calc_hidden(self):
        
        #print("X.shape", self.X.shape, "w_hidden.shape", np.transpose(self.w_hidden).shape)
        hidden_y = self.activation(self.X, np.transpose(self.w_hidden)) # y = f(X*w) <- y = f(v)
        # wx+b <- 
        hidden_x = np.c_[np.ones(len(hidden_y)), hidden_y] # x = [1, y] <-  bias + y 
        y = self.activation(hidden_x, np.array(self.w_output).flatten()) # y = f(x*w) <- y = f(v)
        return hidden_y, hidden_x, y 
    
    def train(self):
        
        #we are going to try to calculate the jacobian matrix while training
        num_samples = len(self.X)  # number of samples (rows) 
        jacobian_size = self.w_hidden.size + self.w_output.size # size of the jacobian matrix (number of weights) + 1 for the bias
        print("jacobian size", jacobian_size)
        jacobian = np.zeros((num_samples, jacobian_size))
        # going for a hard stop on epoch or error
        error = True #error flag (para gui) 
        self.X = np.matrix(self.X) #puntos de entrada (input) (x,y) 
        # weights already initialized
        while self.epoch and error: 
            error = False
            #forward propagation
            hidden_y, hidden_x, y = self.calc_hidden() #calculamos la salida de la capa oculta y la salida de la capa de salida
            # calculate error
            errors = np.array(self.d) - y #calculamos el error (deseado - obtenido) 
            # calculate delta
            delta_output = [] #delta de la capa de salida 
            #back propagation
            for i in range(len(hidden_x)): #para cada punto de entrada (input) 
                dy = self.fd_activation(np.array(y).flatten()[i]) #calculamos la derivada de la funcion de activacion de la capa de salida
                delta_output.append(dy*np.array(errors).flatten()[i]) #this is the gradient of the loss function with respect to the output

                self.w_output = self.w_output + np.dot(hidden_x[i], self.lr*delta_output[-1]) #actualizamos los pesos de la capa de salida (output layer)  lr*delta_output[-1] es el ultimo elemento de la lista delta_output
            for i in range(len(self.X)):
                for j in range(len(self.w_hidden)): 
                    dy = self.fd_activation(hidden_y[i, j])
                    #derivada para delta
                    hidden_delta = np.array(self.w_output).flatten()[j+1]*np.array(delta_output).flatten()[i]*dy # delta = w*delta_output*dy 
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


    def train_jacobian(self):
        
        num_samples = len(self.X) # numero de muestras (filas) es igual al numero de puntos de entrada (input)
        jacobian_size = self.w_hidden.size + self.w_output.size  # el tamaño de la matriz jacobiana es igual al tamaño de los pesos de la capa oculta (hidden layer) + el tamaño de los pesos de la capa de salida (output layer)

        jacobian = np.zeros((num_samples, jacobian_size)) # la matriz jacobiana es una matriz de ceros de tamaño (numero de puntos de entrada por el tamaño de la matriz jacobiana)

        error = True
        self.X = np.matrix(self.X) #puntos de entrada (input) (x,y)
        lambda_factor = 0.1 #factor lambda para el metodo de newton 
        while self.epoch and error: 
            error = False
            #forward propagation
            hidden_y, hidden_x, y = self.calc_hidden() #calculamos la salida de la capa oculta y la salida de la capa de salida
            # calculate error
            errors = np.array(self.d) - y #calculamos el error (deseado - obtenido)
            print("errors", errors)

            for n in range(num_samples): 
                # jacobian matrix filling
                #dL_dy = -2 * (self.d[n]-np.array(y).flatten()[n]) # this is dL/dy which is the derivative of the loss function with respect to output
                dL_dy = -1 
                dy_dv = self.fd_activation(np.array(y).flatten()[n]) # this is dy/dv which is the derivative of the activation function with respect to the output
                dv_dw = hidden_x[n] # this is dv/dw which is the derivative of the output with respect to the weights
                dL_dw = dL_dy*dy_dv*dv_dw# this is the derivative of the loss function with respect to the weights

                # jacobian matrix
                jacobian[n, :self.w_output.size] = dL_dw.flatten() 
                #hidden layer 
                # we need dL_dy, dy_dv

                dv_dh = np.transpose(self.w_output) # this is dv/dh which is the derivative of the output with respect to the hidden layer
                # for each hidden neuron
                for i in range(self.n_neurons):


                    dh_dvh = self.fd_activation(hidden_y[n,i])
                    dvh_dw = self.X[n]

                    dL_dw_hidden = dL_dy*dy_dv*dv_dh[i]*dh_dvh*dvh_dw
                    jacobian[n, self.w_output.size+i*self.w_hidden.shape[1]:self.w_output.size+(i+1)*self.w_hidden.shape[1]] = dL_dw_hidden.flatten()


            jacobian_T = np.transpose(jacobian)

            identity = np.identity(jacobian_T.shape[0])

            gradient_descent = lambda_factor*identity

            hessian = np.dot(jacobian_T, jacobian) + gradient_descent

            hessian_inv = np.linalg.inv(hessian)


            delta =  np.dot(np.dot(hessian_inv, jacobian_T), errors.T)
            delta_1d = np.array(delta).flatten()
            #print("delta1d", delta_1d.shape)

            temp_w_output = self.w_output.copy()
            temp_w_hidden = self.w_hidden.copy()

            square_error = np.average(np.power(errors, 2))


            counter = 0

            for i in range(self.w_output.size):

                self.w_output[i] = self.w_output[i] - delta_1d[i]

                counter += 1
            for i in range(self.n_neurons):

                for j in range(self.X.shape[1]):

                    self.w_hidden[i,j] = self.w_hidden[i,j] - delta_1d[counter]

                    counter += 1
            hidden_y, hidden_x, y_new = self.calc_hidden()
            errors = np.array(self.d) - y_new
            square_error_new = np.average(np.power(errors, 2))
            if square_error_new > square_error:
                square_error = square_error_new
                lambda_factor *= 1.1
            else:
                lambda_factor *= 0.9

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
        _, _, y = self.calc_hidden()

        for i in range(len(np.array(y).flatten())):
            if np.array(y).flatten()[i] >= 0.5:
                self.ax.plot(self.X[i,1], self.X[i,2],marker='o', color='red')
                #self.ax.plot(self.X[i,1], self.X[i,2],'.r')
            else:
                self.ax.plot(self.X[i,1], self.X[i,2],marker='x', color='blue')
                #self.ax.plot(self.X[i,1], self.X[i,2], '.b')

        x_v = np.linspace(-6, 6, 36) #valores de x para el meshgrid 
        y_v = np.linspace(-6, 6, 36)
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

        aX = [-6,6]
        aY = [-6,6]
        self.ax.grid('on')
        self.ax_e.grid('on')
        zeros = np.zeros(2)
        self.ax.plot(aX, zeros, c='k')
        self.ax.plot(zeros, aY, c='k') # k 
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)

    
    def plotting(self,y):
        
        self.ax.cla()

        for i in range(len(np.array(y).flatten())):

            if np.array(y).flatten()[i] >= 0.5:
                #self.ax.plot(self.X[i,1], marker='o', color='red')
                self.ax.plot(self.X[i,1], self.X[i,2], '.r')
            else:
                #self.ax.plot(self.X[i,1], self.X[i,2], marker='o', color='blue')
                self.ax.plot(self.X[i,1], self.X[i,2], '.b')

        x_v = np.linspace(-6, 6, 36) #valores de x para el meshgrid (,36) es el numero de puntos en el meshgrid 
        y_v = np.linspace(-6, 6, 36)
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
