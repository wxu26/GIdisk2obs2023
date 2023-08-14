import numpy as np
from scipy.interpolate import RegularGridInterpolator

class InterpTable():
    def __init__(self):
        self.ndim = 0 # dimension
        self.grid = {} # dictionary, grid_name + grid (1d array)
        self.grid_ind = {} # dictionary, grid_name + dimension (int)
        self.grid_name_list = [] # list of grid_name
        self.grid_finalized = False
        self.data_shape = [] # will be updated with first data
        self.grid_bounds = None # min and max of each grid, will be updated with first data
        self.data = {} # dictionary, data_name + data (ndim d array)
        self.interp_fn = {} # dictionary, data_name + interpolation function
        return
    def __repr__(self):
        """
        print a summary
        """
        s = 'InterpTable object with {} dims\nAxes:'.format(self.ndim)
        for i in range(self.ndim):
            s += '\n  [{}] {}, length={}'.format(i, self.grid_name_list[i], self.data_shape[i])
        s += '\nData fields, shape={}:'.format(self.data_shape)
        for d in self.data:
            s += '\n  ' + d
        return s
    def add_grid(self, grid_name, grid):
        """
        grid_name: string
        grid: 1d array
        """
        if self.grid_finalized:
            raise Exception("Cannot add new grid axis when after grid finalization.")
        if grid_name in self.grid:
            raise Exception("Grid name already exists!")
        self.ndim += 1
        self.grid[grid_name] = grid
        self.grid_ind[grid_name] = self.ndim-1
        self.grid_name_list.append(grid_name)
    def finalize_grid(self):
        self.data_shape = [len(self.grid[grid_name])
            for grid_name in self.grid_name_list]
        self.grid_bounds = np.array([
            [self.grid[grid_name][0],self.grid[grid_name][-1]]
            for grid_name in self.grid_name_list])
        self.grid_finalized = True
    def add_data(self, data_name, fill_val=0.):
        """
        initialize a new data field to zero with the correct shape
        data_name: string
        """
        if data_name in self.data:
            raise Exception("Data name already exists!")
        if not self.grid_finalized: self.finalize_grid()
        self.data[data_name] = np.ones(self.data_shape)*fill_val
    def create_interp_fn(self, data_name):
        grids = [self.grid[grid_name] for grid_name in self.grid_name_list]
        self.interp_fn[data_name] = RegularGridInterpolator(
            grids, self.data[data_name], method='linear', bounds_error=False)
    def create_interp_fn_all_data(self):
        for data_name in self.data:
            self.create_interp_fn(data_name)
        return
    def interp(self, data_name, loglog=False, enforce_bounds=True, **xi):
        """
        perform interpolation
        for performance, there is no error checking
        data_name: name of data field
        loglog: whether the grid/data in this table are log of the actual input/output
        enforce_bounds: whether to bound data to the range of the grid (otherwise oob values become nan)
        **xi: grid name + coordinates of input. must include each and all grid axis
        return: interpolated value
        """
        # check inerpolation function
        if data_name not in self.interp_fn:
            self.create_interp_fn(data_name)
        # load data
        npoint = xi[self.grid_name_list[0]].size
        x = np.zeros((npoint,self.ndim))
        for i in range(self.ndim):
            x[:, i] = xi[self.grid_name_list[i]].flatten()
        # scale
        if loglog: x = np.log(x)
        # apply bounds
        if enforce_bounds:
            x = np.minimum(x, self.grid_bounds[:,1])
            x = np.maximum(x, self.grid_bounds[:,0])
        # interpolate
        y = self.interp_fn[data_name](x)
        # scale
        if loglog: y = np.exp(y)
        return y.reshape(xi[self.grid_name_list[0]].shape)
    def interp_scalar(self, data_name, loglog=False, enforce_bounds=True, **xi):
        """
        a wrapper for interp() for scalar (single-point) input and output
        """
        for grid_name in xi:
            xi[grid_name] = np.array([xi[grid_name]])
        y = self.interp(data_name, loglog=loglog, enforce_bounds=enforce_bounds, **xi)
        return y[0]
    def check_in_bounds(self, loglog=False, **xi):
        """
        check whether the values are in bounds
        """
        for i in range(self.ndim):
            x = xi[self.grid_name_list[i]]
            bounds = self.grid_bounds[i]
            if loglog:
                x = np.log(x)
            if x<bounds[0] or x>bounds[1]:
                print(self.grid_name_list[i], self.grid_bounds[i], x)
                return False
        return True
    def replace_grid(self, grid_name, grid):
        """
        replace a grid and return a new InterpTable object
        all data are mapped onto the new grid
        if grid is a scalar, the corresponding axis is dropped in the new object
        """
        if not self.grid_finalized:
            raise Exception("Grid not finalized!")
        T_new = InterpTable()
        grids = []
        for gn in self.grid_name_list:
            if gn==grid_name:
                if not np.isscalar(grid):
                    T_new.add_grid(gn, grid)
                grids.append(grid)
            else:
                T_new.add_grid(gn, self.grid[gn])
                grids.append(self.grid[gn])
        for dn in self.data:
            T_new.add_data(dn)
        mgrids = np.meshgrid(*grids, indexing='ij')
        xi = {}
        for gn in self.grid_name_list:
            xi[gn] = mgrids[self.grid_ind[gn]]
        for dn in self.data:
            T_new.data[dn] = self.interp(dn, **xi).reshape(T_new.data_shape)
        T_new.create_interp_fn_all_data()
        return T_new




