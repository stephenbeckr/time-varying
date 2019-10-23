function [x,objectiveHist, errHist, str] = firstOrderMethod( ft, gradt, method, tList, lList, muList, varargin )
% [x,objectiveHist, errHist, str] = firstOrderMethod( ft, gradt, method, tList, lList, muList)
%   Runs a first-order method specified by "method"
%       ("method" should be "gradient descent", "nesterov1",
%        "nesterov2","heavyball", or "cg" -- or numbers 1, 2, ... 5)
%       CG stands for Conjugate Gradient
%       Nesterov 1 uses a diminishing coefficient
%       Nesterov 2 takes advantage of knowing the strong convexity constant
%       Gradient descent (and Nesterov) use stepsize 1/L (L=Lipschitz
%           constant of gradient), though 2/(L+mu) gives a better
%           worst-case bound (assumes mu is known)
%
%   This is tailored to TIME-VARYING problems
%       (and may work for generic time-varying problems,
%        but was only tested for quadratics; the CG method
%        is the one that will be most sensitive if not a quadratic)
%
%   ft = @(x,t) returns the objective at x at time t
%   gradt = @(x,t) returns the gradient at x at time t
%
%   Example:
%             A_t     = @(t) A(t:t+T-1,:);
%             y_t     = @(t) y(t:t+T-1);
%             
%             f_t     = @(x,t) norm( A_t(t)*x - y_t(t) )^2/2;
%             grad_t  = @(x,t) A_t(t)'*( A_t(t)*x - y_t(t) );
%
%   tList = [tStart,tEnd] is the start/end time
%       ( if tList is a single number, then we assume tStart=1 )
%       In the static case, this is just an iteration count
%
%   lList is a list of Lipschitz constants of the gradient
%   muList is a list of strong convexity constants
%       (only really needed for Nesterov2 )
%
%  ... = firstOrderMethod( ..., 'parameter', value, ... )
%   allows you to specify more options, like
%       'dimension'     should be the dimension of the primal variable
%       'x0'            is a starting vector
%       'A_t'           is a function @(t) that returns the matrix at time
%           t (only needed for CG)
%       'errFcn'        is a function @(x,t) that returns your choice of
%           an error function (output should be a scalar)
%       'overRelaxation' is between (0,2) [default: 1] and is a step-size
%           multiplier for gradient descent
%       'heavyBallDamping' damps the heavy-ball method, to make it more
%           stable [default: 1]. Any number in (0,1] is valid.
%
% Code: Stephen Becker, October 2019
% Used to make Figure 1 in 
% "Optimization and Learning with Information Streams: Time-varying
% Algorithms and Applications"
% by Emiliano Dall'Anese, Andrea Simonetto, Stephen Becker, Liam Madden
% (Oct 2019, https://arxiv.org/abs/1910.08123 )

prs = inputParser;
addParameter(prs,'dimension',[]);
addParameter(prs,'x0',[]);
addParameter(prs,'overRelaxation', 1); % try 1.9 to be aggressive
addParameter(prs,'stepsizeChoice','avg');
addParameter(prs,'heavyBallDamping',1);
addParameter(prs,'errFcn',[]);
addParameter(prs,'A_t',[]);    % needed only for CG

parse(prs,varargin{:});

p       = prs.Results.dimension;
x       = prs.Results.x0;
if ~isempty(x)
    p = length(x);
else
    if isempty(p)
        error('Must specify dimension or starting point');
    else
        x   = zeros(p,1);
    end
end
stepsizeChoice  = prs.Results.stepsizeChoice;
heavyBallDamping= prs.Results.heavyBallDamping;
errFcn          = prs.Results.errFcn;
overRelaxation  = prs.Results.overRelaxation;
A_t             = prs.Results.A_t;

NESTEROV_StronglyConvex=false;
NESTEROV_Convex=false;
HEAVYBALL=false;
CONJUGATEGRADIENT=false;
switch lower(method)
    case {'gd','gradient descent','gradientdescent',1}
        str     = 'Gradient Descent';
    case {'nesterov1','nesterovv1','nesterov v1','nesterov v.1',2}
        str     = 'Nesterov Acceleration ver 1';
        NESTEROV_Convex = true;
    case  {'nesterov2','nesterovv2','nesterov v2','nesterov v.2',3}
        str     = 'Nesterov Acceleration ver 2';
        NESTEROV_StronglyConvex = true;
    case {'heavy','heavy ball','heavyball','heavy-ball',4}
        str     = 'Heavy-ball method';
        HEAVYBALL = true;
    case {'cg','conjugate gradient',5}
        str     = 'Conjugate gradient';
        CONJUGATEGRADIENT = true;
        if isempty( A_t )
            warning('firstOrderMethod:CG_not_optimal',...
                'For CG, if minimizing a quadratic, need a function that returns the matrix A(t)');
        end
    otherwise
        error('Bad method type: try gd, nesterov1, nesterov2, heavy ball, or cg [or 1,2,3,4,5]');
end
fprintf('Running method %s\n', str );
            
if numel(tList) == 1, tList = [1,tList]; end
if ~isempty( errFcn )
    errHist = zeros( tList(2)-tList(1)+1,1 );
else
    errHist = [];
end
objectiveHist   = zeros( tList(2) - tList(1) + 1, 1 );

z   = x;
xOld= x;


for t = tList(1):tList(end)

    f   = @(x) ft(x,t);
    grad= @(x) gradt(x,t);
    tt  = t - tList(1) + 1;  % make this one 1-based
    
    if length( lList ) == 1 && length( muList ) == 1
        L  = lList;
        mu = muList;
    else
        switch stepsizeChoice
            case 'exact'
                % use a local stepsize
                L       = lList(tt);
                mu      = muList(tt);
            case 'conservative'
                % ... or use conservative stepsize
                L       = max(lList);
                mu      = min(muList);
            case 'avg'
                % ... or use average stepsize
                L       = mean(lList);
                mu      = mean(muList);
            otherwise
                error('StepsizeChoice must be exact/conservative/avg');
        end
    end
    
    
    if NESTEROV_Convex || NESTEROV_StronglyConvex
        xOld    = x;
        x       = z - 1/L*grad(z);
        if NESTEROV_Convex
            z       = x + tt/(tt+3)*(x-xOld);
        elseif NESTEROV_StronglyConvex
            % Constant Step Scheme III
            q_f     = mu/L;
            z       = x + (1-sqrt(q_f))/(1+sqrt(q_f))*(x-xOld);
        end
    elseif HEAVYBALL && tt > 1
        LL      = sqrt(L);
        mm      = sqrt(mu);
        alpha   = heavyBallDamping*4/( (LL+mm)^2 );
        beta    = heavyBallDamping*( (LL-mm)/(LL+mm) )^2; % equiv, ( ( sqrt(q_f)-1 )/(sqrt(q_f)+1) )^2
        z       = x;
        x       = x - alpha*grad(x) + beta*(x-xOld);
        xOld    = z;
    elseif CONJUGATEGRADIENT % adding June 2019
        % Find stepsize alpha, based on x and direction pCG
        % r = gradient
        % Exact line search (for static, quadratic case)
        gradOld = grad(x);
        if tt==1
            pCG     = -gradOld;
        end
        % Use optimal stepsize (closed form, since quadratic objective)
        if isempty( A_t )
            alphak = 1/L;   % not a quadratic, so use simple stepsize
        else
            alphak  = -gradOld'*pCG/( norm(A_t(t)*pCG)^2 ); % Optimal even in time-varying case!
        end
        
        x       = x + alphak*pCG;
        
        gradNew = grad(x);
%         beta  = (norm(gradNew)/norm(gradOld))^2; % Fletcher-Reeves
%         beta  = (gradNew'*( gradNew-gradOld)/norm(gradOld))^2; % Polak-Ribiere
%         beta  = max(0,(gradNew'*( gradNew-gradOld)/norm(gradOld))^2); % Polak-Ribiere+
        beta = norm(gradNew)^2/( (gradNew-gradOld)'*pCG ); % eq 5.49, Nocedal & Wright
        pCG     = -gradNew + beta*pCG;
    else
        xOld    = x; % needed for HeavyBall
        x       = x - overRelaxation/L*grad(x);
    end

    if ~isempty( errFcn )
        errHist(tt)           = errFcn(x,t);
    end
    objectiveHist(tt)   = f(x);
end
