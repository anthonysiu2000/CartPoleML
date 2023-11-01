import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost. 
        Parameter:
            s (1D numpy array) with shape (4,) 
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation


    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy array with shape (1,)
                such that the optimial policies at time are i is K_i * x_i + k_i
                where x_i is the state
        """
        #TODO
        
        n_s = len(s_star)
        n_a = len(a_star)
        
        #The following lamda functions will get the corespoinding f or c functions that utilize either s_star or a_star as the input values
        fwiths = lambda s_star: self.f(s_star, a_star)
        fwitha = lambda a_star: self.f(s_star, a_star)
        cwiths = lambda s_star: self.c(s_star, a_star)
        cwitha = lambda a_star: self.c(s_star, a_star)
        
        
        #A is the jacobian of f, with respect to s
        A = jacobian(fwiths, s_star)
        #B is the jacobian of f, with respect to a
        B = jacobian(fwitha, a_star)
        
        #Q is the hessian of c, with respect to s twice
        Q = hessian(cwiths, s_star)
        #regularize Q
        Q = Q + 1e-4 * np.eye(n_s)
        
        
        #R is the hessian of c, with respect to a twice
        R = hessian(cwitha, a_star)
        #regularize R
        R = R + 1e-4 * np.eye(n_a)
        
        #M is the hessian of c, with respect to s and a
        f = lambda s_star: self.c(s_star, a_star)
        grad = lambda a_star: gradient(f, s_star)
        M = jacobian(grad, a_star)
        
        #q is the gradient of c, with respect to s
        q = gradient(cwiths, s_star)
        #reshape into 2d
        q = q.reshape(n_s,1)
        
        #r is the gradient of c, with respect to a
        r = gradient(cwitha, a_star)
        #reshape into 2d
        r = r.reshape(n_a,1)
        
        
        #m, according to section 4.4
        m = self.f(s_star, a_star) - A @ s_star - B @ a_star
        #reshape into 2d
        m = m.reshape(n_s,1)
        
        #b, according to section 4.4
        b = self.c(s_star, a_star) + 1/2 * s_star.T @ Q/2 @ s_star + 1/2 * a_star.T @ R @ a_star + s_star.T @ M @ a_star - q.T @ s_star - r.T @ a_star.T
        
        
        out = lqr(A, B, m, Q, R, M, q, r, b, T)

        return out
class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a



