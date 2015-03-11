"""Line searchf
"""

import numpy as np
import numbers


__all__=[
    "BackTrackingLineSearch"     
]


class BackTrackingLineSearch(object):
    """Back tracking inexact line search"""    
    def __init__(self):        
        pass    

    
    def search(self,
               objective_function = None,
               start_point = None,
               gradient_at_start_point = None,
               search_direction = None, 
               step_size_shrink_factor = 0.9,
               stop_condition_control_parameter = 0.5):
        """
        Parameters:
        -----------
        objective_function: callable object
            Objective funciton to be minimized
            
        start_point: 1D array
            point that we start to do line search.  
            
        gradient_at_start_point: 1D array 
            gradient of objective function evalued at the start point            
            
        search_direction: 1D array
            direction along which searching will be carried out
        
        step_size_shrink_factor: float. (0, 1) default 0.9
            control the speed of the shrinkage of the step size.

        stop_condition_control_parameter: (0, 0.5] default 0.5           
            control parameter used in Armijo-Goldstein condition
        
        Return:
        -------
        t: float
            optimal step size x^{*} = x + t * search_direction
            
        iter: int 
            total iteration that has been carried out
        """
        
        # Check parameters
        if not hasattr(objective_function, '__call__'):
            raise ValueError(
                "objective_function must be callable"
                )       
        
        if not isinstance(start_point, (np.ndarray)):
            raise ValueError(
                "start_point must be 1D array"
                )
            
        if not isinstance(gradient_at_start_point, (np.ndarray)):
            raise ValueError(
                "gradient_at_start_point must be 1D array"
                )            
        
        if not isinstance(search_direction, (np.ndarray)):
            raise ValueError(
                "search_direction must be 1D array"
                )
        
        if (not isinstance(step_size_shrink_factor, numbers.Number) or 
                step_size_shrink_factor >= 1.0 or step_size_shrink_factor <= 0.0):
            raise ValueError(
                "step_size_shrink_factor must be number within (0, 1)"
                )

        if (not isinstance(stop_condition_control_parameter, numbers.Number) or 
                stop_condition_control_parameter > 0.5 or stop_condition_control_parameter <= 0.0):
            raise ValueError(
                "stop_condition_control_parameter must be number within (0, 0.5]"
                )                
        
            
        # parameters used by back tracking line search
        alpha = stop_condition_control_parameter  # (0, 0.5)
        beta = step_size_shrink_factor            # (0, 1)    
        
        # Input 
        f = objective_function
        x = start_point
        dx = search_direction
        df_x = gradient_at_start_point
        
        t = 1
        iter = 0
        while f(x+t*dx) >= f(x) + alpha * t * np.dot(df_x, dx):  # Wolfe condition
            t *= beta
            iter += 1
             
        return t, iter
        
        