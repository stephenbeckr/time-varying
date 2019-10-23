%{
Recreates Figure 1 from:

"Optimization and Learning with Information Streams: Time-varying
Algorithms and Applications"
by Emiliano Dall'Anese, Andrea Simonetto, Stephen Becker, Liam Madden
(Oct 2019, https://arxiv.org/abs/1910.08123 )

This is a least-squares problem in the static case,
 and in the time-varying/dynamic case, we use a slide window for
 the data for the least-squares problem

Relies on the code "firstOrderMethod.m" to run the algorithms

Stephen Becker, 2019
%}

% -- Setup problem data --
rng(0);
p   = 50;
N   = 1e3;
T   = p; % try this, since I wasn't getting good story with T=1e2
A   = randn(N,p);
% Let's add in 2 "jumps" and use this to modify the right-hand-side "y"
xFull1  = rand(p,1);
xFull2  = xFull1 + .05*rand(p,1);
xFull3  = xFull1 + .05*rand(p,1);
y1   = A*xFull1 + .01*randn(N,1)/sqrt(N);
y2   = A*xFull2 + .01*randn(N,1)/sqrt(N);
y3   = A*xFull3 + .01*randn(N,1)/sqrt(N);
y   = [ y1(1:299); y2(300:599); y3(600:end) ];
% So, we have problem data (A,y), and look at sliding time windows of this

tList  = 1:N-T+1;   % times

% Compute Lipshitz constants
[lList,muList] = deal( zeros(length( tList ),1) );
for t = tList
    AT  = A(t:t+T-1,:);     % this is size T x p
    yT  = y(t:t+T-1);       % this is size T x 1
    
    G   = AT'*AT;
    e   = eig(G);
    lList(t)   = max(e);    % Lipschitz constant of gradient
    muList(t)  = min(e);    % strong convexity constant

end
L_global    = norm(A)^2;

A_t     = @(t) A(t:t+T-1,:);
y_t     = @(t) y(t:t+T-1);

f_t     = @(x,t) norm( A_t(t)*x - y_t(t) )^2/2;
grad_t  = @(x,t) A_t(t)'*( A_t(t)*x - y_t(t) );
x_t     = A_t(t)\y_t(t); % true answer, if needed
errFcn  = @(x,t) norm( x - x_t(t) );

%% Static case

maxIters   = 3e3; % number of iterations
tFixed  = 2;
f       = @(x,ignore) f_t(x,tFixed);
grad    = @(x,ignore) grad_t(x,tFixed);
L       = lList(tFixed);
mu      = muList(tFixed);

figure(1); clf;
for method  = 1:5
    [x,objectiveHist, errHist, str] = firstOrderMethod( f, grad, method, ...
        maxIters, L, mu, 'x0',zeros(p,1), 'A_t', @(ignore) A_t(tFixed) );

    semilogy( objectiveHist, 'linewidth',2,'DisplayName',str);
    hold all
end
set(gca,'fontsize',18)
ylabel('Sub-optimality of objective function');
xlabel('Iterations');
legend;
title('Static Case');
ylim([1e-15,1e5]);
%% Dynamic case

figure(2); clf;
for method  = 1:5
    [x,objectiveHist, errHist, str] = firstOrderMethod( f_t, grad_t, method, ...
        tList, lList, muList,'stepsizeChoice','avg','x0',zeros(p,1),...
        'A_t', A_t );
    semilogy( objectiveHist, 'linewidth',2,'DisplayName',str);
    hold all
end
set(gca,'fontsize',18)
ylabel('Sub-optimality of objective function');
xlabel('Time index');
legend;
title('Dynamic Case');
ylim([-Inf,1e4]);