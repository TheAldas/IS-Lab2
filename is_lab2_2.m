clear; clc; close all;

% Duomenys
x = linspace(0.1,1,20);
x_test = linspace(0.1, 1, 330);
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;  % norimas atsakas

X = x(:); 
T = d(:);

% struktura
nHidden = 6;      
lr = 0.03;
epochs = 50000;

W1 = randn(nHidden,1);
b1 = randn(nHidden,1);
W2 = randn(1,nHidden);
b2 = randn(1,1);

N = length(X);

% Mokymas
mse = zeros(epochs,1);

for epoch = 1:epochs
    err_total = 0;
    
    for i = 1:N
        xi = X(i);
        ti = T(i);
        
        % Atsakas
        a_hidden = zeros(nHidden,1);
        for h = 1:nHidden
            z = W1(h)*xi + b1(h);
            a_hidden(h) = tanh(z);
        end
        
        y_hat = 0;
        for h = 1:nHidden
            y_hat = y_hat + W2(h)*a_hidden(h);
        end
        y_hat = y_hat + b2;
        
        % Palyginimas
        e = ti - y_hat;
        err_total = err_total + e;
        
        % Atnaujinimas
        dW2 = zeros(1,nHidden);
        for h = 1:nHidden
            dW2(h) = e * a_hidden(h);
        end
        db2 = e;
        
        % Paslepto sluoksnio atnaujinimas
        dW1 = zeros(nHidden,1);
        db1 = zeros(nHidden,1);
        for h = 1:nHidden
            delta_h = (1 - a_hidden(h)^2) * (W2(h) * e);
            dW1(h) = delta_h * xi;
            db1(h) = delta_h;
        end
        
        % Atnaujinimas
        for h = 1:nHidden
            W2(h) = W2(h) + lr*dW2(h);
        end
        b2 = b2 + lr*db2;
        
        for h = 1:nHidden
            W1(h) = W1(h) + lr*dW1(h);
            b1(h) = b1(h) + lr*db1(h);
        end
    end
end

% Rezultatai
Yhat = zeros(N,1);
for i = 1:N
    a_hidden = zeros(nHidden,1);
    for h = 1:nHidden
        z = W1(h)*X(i) + b1(h);
        a_hidden(h) = tanh(z);
    end
    
    y_hat = 0;
    for h = 1:nHidden
        y_hat = y_hat + W2(h)*a_hidden(h);
    end
    Yhat(i) = y_hat + b2;
end

figure;
plot(X,T,'bo-','LineWidth',1.5); hold on;
plot(X,Yhat,'r*-','LineWidth',1.5);
legend('Norimas atsakas','MLP aproksimacija');
xlabel('x'); ylabel('y');
title('MLP su 6 paslėptais neuronais (rankinė realizacija)');
grid on;
