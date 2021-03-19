import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import time

def getData():
	try:
		with gzip.open("data.pickle.gz",'rb') as FILE:
			data=pickle.load(FILE)

	except Exception as e:
		print(e)
		print('LOADING DATA ONLINE')
		#TENSORFLOW FOR LOADING DATA
		import tensorflow as tf
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		data = (x_train, y_train), (x_test, y_test) 
		with gzip.open("data.pickle.gz",'wb') as FILE:
			pickle.dump(data,FILE)
	return data

def one_hot_vector(x):
	I = np.eye(10)
	return I[x]

def preprocesing(data):
	(x_train, y_train), (x_test, y_test)= data

	#NORMALIZATION
	x_train = x_train/255
	x_test  = x_test/255

	#FlATING IMAGES
	x_train = np.array([item.reshape(784) for  item in x_train])
	x_test  = np.array([item.reshape(784) for  item in x_test])


	#ONE HOT ENCODING 
	y_train = np.array([one_hot_vector(i) for i in y_train])
	y_test  = np.array([one_hot_vector(i) for i in y_test])

	return x_train, y_train, x_test, y_test



data= getData()

x_train, y_train,x_test, y_test = preprocesing(data)




class Linear:
	def __init__(self,input_shape,output_shape,activation=None):
		self.weights= (   
						  np.random.rand(input_shape,output_shape)     #random weights init
						- np.random.rand(input_shape,output_shape)
					  ) 
		self.biases = np.zeros((1,output_shape))                    ##random biases init
		self.activation=activation

	def forward(self,inputs):
		self.inputs =  inputs
		self.output =self.activation.forward(  np.dot(inputs,self.weights) 
											  + self.biases
											)
		return self.output




class Sigmoid:

	def forward(self,x):
		self.output = 1/(1+np.exp(-x)) 
		return self.output

	def derivative(self):
		return np.multiply(self.output,(1-self.output))






class NeuralNetwork(object):
	def __init__(self):
		self.layers=[Linear(28*28,64, activation=Sigmoid()), # input->784 (28*28 , IMG_SIZE) shape of image out-> 64
					 Linear(64,64, activation=Sigmoid()),
					 Linear(64,10, activation=Sigmoid()),] # 784 shape of image


		self.learning_rate = 0.02

	def forward(self,x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backword(self,inputs,y):
		pred = self.forward(inputs)
		error = (y-pred)*2/len(y[0])

		for i in range(len(self.layers)-1,-1,-1):
			dJdA = error
			dAdZ = self.layers[i].activation.derivative()
			dJdW = self.layers[i].inputs

			dJdZ = dJdA * dAdZ
			dJdW = np.dot( dJdW.T , dJdZ )

			self.layers[i].weights +=  dJdW * self.learning_rate
			self.layers[i].biases  +=  np.sum(dJdZ ,axis=0) * self.learning_rate


			error =np.dot(dJdZ,self.layers[i].weights.T)
		

	
	def shuffle_data(self,x,y):
		arr=np.arange(len(x))
		np.random.shuffle(arr)
		new_x,new_y=[],[]

		for i in arr:
			new_x.append(x[i])
			new_y.append(y[i])

		return np.array(x) , np.array(y)

	def loss(self,pred,y):
		loss = np.sum((pred-y)**2 / len(pred[0]))
		return loss
	
	def train(self,x,y,batch_size=16,epochs=1,test_data=None):
		batch_size=min(len(x),batch_size)
		x,y = self.shuffle_data(x,y)
		DATA_LEN = len(x)
		if test_data :
			x_test ,y_test = test_data

		for epoch in range(epochs):
			split=0
			while split<DATA_LEN:
				#Split Data into Batches
				batch_x=x[split:split+batch_size]
				batch_y=y[split:split+batch_size]

				self.backword(batch_x,batch_y)
				split+=batch_size
			loss=0
			for i in range(10):
				n=np.random.randint(0,DATA_LEN-1)
				pred = self.forward(x[n:n+batch_size])
				y_=y[n:n+batch_size]
				loss+=self.loss(pred,y_)



			print(f"Loss of epoch {epoch}: {loss/10}  ")

		self.learning_rate*=0.8





Net= NeuralNetwork()

t=time.time()

Net.train(x_train,y_train,batch_size=16,epochs=10)

print(f"Time to train {time.time()-t}")

count=0
for n in range(len(x_test)):
	if np.argmax(Net.forward(np.array([x_test[n]]))[0])==np.argmax((y_test[n])):
		count+=1

print(f"Test Accuray:{'%.2f'%float((count/len(x_test)) *100)}")

for _ in range(10):
	n=np.random.randint(0,len(x_test)-1)
	print(f"Pred {np.argmax(Net.forward(np.array([x_test[n]]))[0])} {np.argmax((y_test[n]))}")
	plt.imshow(x_test[n].reshape(28,28))
	plt.show()