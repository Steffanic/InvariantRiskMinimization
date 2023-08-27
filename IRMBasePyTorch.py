import torch
from torch.autograd import grad
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def compute_penalty(losses,dummy_w):
    g1=grad(losses[0::2].mean(),dummy_w,create_graph=True)[0] # minibatch 1
    g2=grad(losses[1::2].mean(),dummy_w,create_graph=True)[0] # minibatch 2
    return(g1*g2).sum() 

def example_1(n=10000,d=2,env=1): 
    x=torch.randn(n,d)*env 
    y=x+torch.randn(n,d)*env 
    z=y+torch.randn(n,d) 
    return torch.cat((x,z),1), y.sum(1,keepdim=True) 

def gaussian_with_different_noise(n=10000,d=2,env=1):
    x = torch.randn(n,d)
    y = x + torch.randn(n,d)*env
    return x, y

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = torch.from_numpy(x_train).float().view(-1, 28*28)
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float().view(-1, 28*28)
    y_test = torch.from_numpy(y_test).long()
    return x_train, y_train, x_test, y_test

phi=torch.nn.Parameter(torch.ones(28*28, 10)) 
dummy_w=torch.nn.Parameter(torch.Tensor([1.0])) 
opt=torch.optim.SGD([phi],lr=1e-3) 
scce=torch.nn.CrossEntropyLoss(reduction="none")
softmax=torch.nn.Softmax(dim=1)
#mse=torch.nn.MSELoss(reduction="none") 
#environments=[example_1(env=1.0), example_1(env=0.9), example_1(env=1.1)] 
environments = []



for iteration in range(50000): 
    print(f"Iteration {iteration}")
    error=0 
    penalty=0 
    for x_e,y_e in environments: 
        p=torch.randperm(len(x_e)) 
        logits=softmax(x_e[p]@phi)
        error_e=scce(logits*dummy_w, y_e[p]) 
        penalty+=compute_penalty(error_e,dummy_w) # IRM term
        error+=error_e.mean() # ERM term
        opt.zero_grad() 
        (1e-4*error+penalty).backward(retain_graph=True) 
        opt.step() 
    print(error.item(),penalty.item())
    if iteration%100==0: 
        plt.figure(figsize=(10, 4))
        for digit in range(10):
            plt.subplot(2, 5, digit+1)
            plt.matshow(phi.detach().numpy()[:,digit].reshape(28,28), fignum=0)
            plt.title(f"Digit {digit}, Iteration {iteration}")

        plt.show()
        
        # evaluate on test set
        test_logits = softmax(x_test@phi)
        test_error = scce(test_logits*dummy_w, y_test).mean()
        print(f"Test error: {test_error.item()}")
        # also calculate the accuracy 
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test).float().mean()
        print(f"Test accuracy: {test_acc.item()}")
        # evaluate on train set
        train_logits = softmax(x_train@phi)
        train_error = scce(train_logits*dummy_w, y_train).mean()
        print(f"Train error: {train_error.item()}")
        # also calculate the accuracy
        train_preds = torch.argmax(train_logits, dim=1)
        train_acc = (train_preds == y_train).float().mean()
        print(f"Train accuracy: {train_acc.item()}")
        print("")

