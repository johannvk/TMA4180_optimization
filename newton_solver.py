# optimizing using gradient descent and newton
import numpy as np
#import np.linalg as la


def find_step_direction(f,df,xk,B,P):
    return np.linalg.solve(B(P,xk,l),-df(P,xk,l))

def find_minimum(f, P,df,theta_0, B, tol=1e-6, max_iter=50, c1=0.5, c2=0.8):
    # f is function to be minimized
    # P is point to be reached
    # df is "nabla"f (a function)
    # B(xk) is a function returning Bk*xk,
    # where Bk is the matrix used in defining the direction vector p in step k.
    # Gradient descent method is Bk = Id. Newtons method uses Bk = hessian(fk).
    i=0
    theta_k = theta_0
    improvement = tol+1
    while i<max_iter and improvement>tol:
        i+=1
        p=find_step_direction(f,df,theta_k,B,P)
        a_max = 1e100 # initial step length
        a_min = 0
        a = 1
        j=0
        while j<max_iter:
            j+=1
            if f(P, theta_k + a*p,l) > f(P,theta_k,l) + a*c1*df(P,theta_k,l)@p:
                a_max = a
                a = a_max/2 + a_min/2
            elif df(P,theta_k + a*p,l).T@p <= c2*df(P,theta_k,l).T@p:
                a_min = a
                if a_max > 1e99:
                    a*=2
                else:
                    a = (a_max + a_min)/2
            else:
                break
        improvement = f(P,theta_k,l) - f(P,theta_k + a*p,l)
        print("improvement=",improvement)
        theta_k = theta_k + a*p
    print("Number of steps=",i)
    return theta_k, f(P,theta_k,l)

def x_fnc(theta, l):
    x=0
    for i, li in enumerate(l):
        phi = 0
        for j in range(i+1):
            phi += theta[j]
        x += np.cos(phi)
    return x

def dx(k,theta,l):
    return -y_fnc(theta[k:],l[k:])

def ddx(k,m,theta,l):
    return -x_fnc(theta[max(k,m):],l[max(k,m):])

def y_fnc(theta, l):
    y=0
    for i, li in enumerate(l):
        phi = 0
        for j in range(i+1):
            phi += theta[j]
        y += np.sin(phi)
    return y

def dy(k,theta,l):
    return x_fnc(theta[k:],l[k:])

def ddy(k,m,theta,l):
    return -y_fnc(theta[max(k,m):],l[max(k,m):])

def f(p,theta,l):
    return (p[0]-x_fnc(theta,l))**2 + (p[1]-y_fnc(theta,l))**2

# Should be made more effective
def df(p,theta,l):
    Df = []
    for k,L in enumerate(l):
        Df.append(- 2*(p[0]-x_fnc(theta,l))*dx(k,theta,l) - 2*(p[1]-y_fnc(theta,l))*dy(k,theta,l))
    return np.array(Df)

# Should be made more effective
def ddf(p,theta,l):
    DDf=[]
    for k,L in enumerate(l):
        row=[]
        for m,L in enumerate(l):
            row.append(2*(x_fnc(theta,l)-p[0])*ddx(k,m,theta,l) + 2*dx(k,theta,l)*dx(m,theta,l) + 2*(y_fnc(theta,l)-p[1])*ddx(k,m,theta,l) + 2*dy(k,theta,l)*dy(m,theta,l))
        DDf.append(row)
    return np.array(DDf)


print("problem 1")
n=3
theta_0 = np.zeros(n)
l=np.array([3,2,2])
p=(3,2)
print("theta, f(theta)=",find_minimum(f,p,df,theta_0, ddf))

print("\nproblem 2")
n=3
theta_0 = np.zeros(n)
l=np.array([1,4,1])
p=(1,1)
print("theta, f(theta)=",find_minimum(f,p,df,theta_0, ddf))

print("\nproblem 3")
n=4
theta_0 = np.zeros(n)
l=np.array([3,2,1,1])
p=(3,2)
print("theta, f(theta)=",find_minimum(f,p,df,theta_0, ddf))

print("\nproblem 4")
p=(0,0)
print("theta, f(theta)=",find_minimum(f,p,df,theta_0, ddf))
