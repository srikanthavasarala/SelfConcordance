clc
clear


randn('state',1);
m=10000; % number of samples of a_i to generate
n=1000; % in R^n
alpha = 0.01; % Backtracking parameters
beta = 0.5; % " " "
iteration_tolerance = 1000;
error_tolerance = 1e-8; % epsilon tolerance of error


% generate random problem
ai_matrix = randn(m,n);% generate samples of a_i, i = 1...m

% Here begins the Newton's method with backtracking
x_iterateset = []; number_iterations = [];
x = zeros(n,1); % initial x_0 = 0
f_x0 = -sum(log(1-ai_matrix*x)) - sum((log(1-x.^2)));
for iter = 1:iteration_tolerance
iter    
f_x = -sum(log(1-ai_matrix*x)) - sum((log(1-x.^2))); % Computing f at x
x_iterateset = [x_iterateset, f_x];
d_logx = 1./(1-ai_matrix*x);
gradient_f = (ai_matrix')*d_logx - 1./(1+x) + 1./(1-x); % grad f at x
hessian_f = ai_matrix'*diag(d_logx.^2)*ai_matrix + diag(1./(1+x).^2 + 1./(1-x).^2); %Hessian of f at x
v = -hessian_f\gradient_f; % Newton search direction
lambda_square = gradient_f'*v % lambda^2 (x) Newton decrement
if abs(lambda_square) < error_tolerance, break; end;
t = 1;
while ( -sum(log(1-ai_matrix*(x+t*v))) - sum(log(1-(x+t*v).^2)) > ...
f_x + alpha*t*lambda_square ) % backtracking condition
t = beta*t;
end;
x = x+t*v;
number_iterations = [number_iterations,t];
end;

optimum_value = x_iterateset(length(x_iterateset));
selfconcordant_bound = ((20-8*alpha)/(alpha*beta*(1-2*alpha)^2))*(f_x0 - optimum_value) + log2(log2(1/error_tolerance));
%% 
saveid_png = 'ConvergenceConcordant';
figure; % log(absolute error) vs number of steps plot
 semilogy([0:(length(x_iterateset)-2)], x_iterateset(1:length(x_iterateset)-1)-optimum_value, '-', ...
 [0:(length(x_iterateset)-2)], x_iterateset(1:length(x_iterateset)-1)-optimum_value, 'o','linewidth',1.5);
xlabel('Number of iterations'); ylabel('Absolute error'); grid on;


% export_fig(gcf,'-dpng','-r300',fullfile('/home/srikanth/Documents/Courses/CVO/ProjectCode',saveid_png));

