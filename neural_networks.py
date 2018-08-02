# ACTIVATION FUNCTIONS
def euler(x):
    return 2.718281828459045235360287471352662 ** max(min(x,20),-20)
        
def relu(x):
    return max(0.,x)

def sigmoid(x):
    return 1/(1 + euler(-x))

def tanh(x):
    a,b = euler(x),euler(-x)
    return (a-b)/(a+b)

def phi(ys,activation=relu):
    return [[activation(y) for y in y_row] for y_row in ys]

# GRADIENTS / BACKPROP
def error(y_desired,y_predicted):
    return y_desired - y_predicted

def loss(y_desired, y_predicted):
    return (error(y_desired,y_predicted)**2) / 2

def dSigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def updates(ws,dWs,learning_rate = .001):
    return ws + learning_rate * (- dWs)

# MATRIX METHODS
def transpose(xs):    
    return [[x_row[i] for x_row in xs] for i in range(len(xs[0]))]

def dot(xs,ws):
    return [[sum([x*w for x,w in zip(row_x,row_w)]) for row_w in transpose(ws)] for row_x in xs ]

def plus(xs,bs):
    return [[x+b for x,b in zip(x_row,b_row)] for x_row,b_row in zip(xs,bs)]
'''
def efficient_dot(xs,ws):
    ys = [0 for _ in range(len(ws[0]))]
    indices = [index for index,x in enumerate(xs) if x != 0]
    for i in range(len(ys)):
        for index in indices:
            ys[i] += ws[index][i]
    return ys
'''

# ANN
def forward(xs,ws,bs,activation=relu):
    return phi(plus(dot(xs,ws),bs),activation)

def xor_ffnn(a,b):
    xs = [[a,b]]
    ws_x = [[2,-1],[-1,2]]
    bs_x = [[0,0]]
    hs = forward(xs,ws_x,bs_x,tanh)
    ws_h = [[2],[2]]
    bs_h = [[1]]
    ys = forward(hs,ws_h,bs_h,sigmoid)
    return ys[0][0]

print(xor_ffnn(0,0))
