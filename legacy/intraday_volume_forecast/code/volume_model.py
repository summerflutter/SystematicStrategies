import numpy as np
import pandas as pd

class VolumeModel:
    def __init__(self, fixed_pars=None, init_pars=None):
        self.par = {
            "a_eta": np.nan,
            "a_mu": np.nan,
            "var_eta": np.nan,
            "var_mu": np.nan,
            "r": np.nan,
            "phi": np.nan,
            "x0": np.array([np.nan, np.nan]),
            "V0": np.array([np.nan, np.nan, np.nan])
        }
        
        self.init = {}
        self.converged = {}
        
        if fixed_pars is not None:
            self._check_pars_list(fixed_pars, init_pars)
            self._update_fixed_pars(fixed_pars)

        if init_pars is not None:
            self._update_init_pars(init_pars)

        self._check_convergence()

    def _check_pars_list(self, fixed_pars, init_pars):
        """
        Ensure the proper structure of parameters.
        """
        if not isinstance(fixed_pars, dict):
            raise TypeError("fixed_pars must be a dictionary.")
        if not isinstance(init_pars, dict):
            raise TypeError("init_pars must be a dictionary.")
        
        # Additional checks on length and contents of init_pars and fixed_pars
        for key, value in fixed_pars.items():
            if key not in self.par:
                raise ValueError(f"Fixed parameter {key} is not valid.")
            if not np.isfinite(value):
                raise ValueError(f"Value for {key} must be a finite number.")

        for key, value in init_pars.items():
            if key not in self.par:
                raise ValueError(f"Initial parameter {key} is not valid.")
            if not np.isfinite(value):
                raise ValueError(f"Value for {key} must be a finite number.")

    def _update_fixed_pars(self, fixed_pars):
        """
        Updates the fixed parameters in the model.
        """
        for name, value in fixed_pars.items():
            self.par[name] = value
    
    def _update_init_pars(self, init_pars):
        """
        Updates the initial parameters in the model.
        """
        for name, value in init_pars.items():
            self.init[name] = value

    def _check_convergence(self):
        """
        Check whether all parameters have been properly initialized.
        """
        for name in self.par:
            self.converged[name] = not np.isnan(self.par[name])

    def __repr__(self):
        return f"VolumeModel(par={self.par}, init={self.init}, converged={self.converged})"
    
    def warnings(self):
        """
        Generates warnings if any parameters are improperly set.
        """
        warnings = []
        if any(np.isnan(val) for val in self.par.values()):
            warnings.append("Some parameters in 'par' are not initialized.")
        
        if any(np.isnan(val) for val in self.init.values()):
            warnings.append("Some parameters in 'init' are not initialized.")
        
        return warnings


# Example of using the VolumeModel
fixed_pars = {
    "a_eta": 0.5,
    "a_mu": 0.3,
    "var_eta": 0.1,
    "var_mu": 0.2,
    "r": 0.99,
    "phi": 0.9,
    "x0": np.array([0.0, 1.0]),
    "V0": np.array([0.1, 0.1, 0.1])
}

init_pars = {
    "r": 0.98,
    "phi": 0.85
}

vm = VolumeModel(fixed_pars=fixed_pars, init_pars=init_pars)

# Check for warnings
warnings = vm.warnings()
if warnings:
    for warning in warnings:
        print(warning)

# Display the model
print(vm)



def clean_data(data: pd.DataFrame) -> np.ndarray:
    """
    Remove trading days (columns) with any NA.
    If index is datetime, assume it's xts-style intraday data, and unify as matrix.
    """
    if isinstance(data.index, pd.DatetimeIndex):
        return intraday_xts_to_matrix(data)
    else:
        cols_with_na = data.columns[data.isna().any()]
        data_cleaned = data.drop(columns=cols_with_na)
        if len(cols_with_na) > 0:
            msg = (
                "For input matrix:\n"
                f" Remove trading days with missing bins: {', '.join(map(str, cols_with_na))}.\n"
            )
            warnings.warn(msg)
        return data_cleaned.to_numpy()


def intraday_xts_to_matrix(data_xts: pd.DataFrame) -> np.ndarray:
    """
    Remove any day with NA or incomplete bin count.
    Convert xts-style data (datetime-indexed intraday data) into matrix.
    """
    # Step 1: remove days with any NA
    data_xts = data_xts.copy()
    daily_has_na = data_xts.groupby(data_xts.index.date).apply(lambda df: df.isna().any().any())
    days_no_na = [day for day, has_na in daily_has_na.items() if not has_na]
    
    # Filter only days without NA
    data_xts = data_xts[data_xts.index.date.astype(str).isin([str(d) for d in days_no_na])]

    # Step 2: find max bin count and keep only full-bin days
    bins_count = data_xts.groupby(data_xts.index.date).size()
    max_bin = bins_count.max()
    full_bin_days = bins_count[bins_count == max_bin].index

    # Final filter: keep only good days
    valid_days = sorted(set(days_no_na).intersection(set(full_bin_days)))
    data_xts = data_xts[data_xts.index.date.astype(str).isin([str(d) for d in valid_days])]

    # Convert to matrix
    data_mat = data_xts.to_numpy().reshape((max_bin, -1), order='F')

    # Step 3: Warnings for dropped days
    dropped_na_days = set(daily_has_na[daily_has_na].index)
    dropped_incomplete_days = set(bins_count[bins_count != max_bin].index)
    dropped_days = sorted(dropped_na_days.union(dropped_incomplete_days))

    if dropped_days:
        msg = (
            "For input xts:\n"
            f" Remove trading days with missing bins: {', '.join(map(str, dropped_days))}.\n"
        )
        warnings.warn(msg)

    return data_mat



import numpy as np

def clean_pars_list(input_dict):
    all_pars_name = {"a_eta", "a_mu", "var_eta", "var_mu", "r", "x0", "V0", "phi"}
    expected_pars_len = {
        "a_eta": 1, "a_mu": 1,
        "var_eta": 1, "var_mu": 1,
        "r": 1,
        "x0": 2, "V0": 4
    }

    invalid_param = []
    incorrect_param = []
    clean_dict = {}

    for key, val in input_dict.items():
        if key not in all_pars_name:
            invalid_param.append(key)
            continue

        arr = np.atleast_1d(val)
        if not np.issubdtype(arr.dtype, np.number) or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            incorrect_param.append(key)
            continue

        if key == "phi":
            clean_dict[key] = arr
            continue

        if key in expected_pars_len and len(arr) != expected_pars_len[key]:
            incorrect_param.append(key)
            continue

        if key == "V0":
            V_matrix = np.array([[arr[0], arr[1]], [arr[1], arr[3]]])
            eigvals = np.linalg.eigvalsh(V_matrix)
            if not np.allclose(V_matrix, V_matrix.T) or np.any(eigvals < 0):
                incorrect_param.append(key)
                continue
            clean_dict[key] = arr[[0, 1, 3]]  # keep 3 elements

        elif key == "r":
            if arr[0] < 0:
                incorrect_param.append(key)
                continue
            clean_dict[key] = arr[0]

        else:
            clean_dict[key] = arr if len(arr) > 1 else arr[0]

    messages = []
    if invalid_param:
        messages.append(f"Elements {', '.join(invalid_param)} are not allowed in parameter list.")
    if incorrect_param:
        messages.append(f"Elements {', '.join(set(incorrect_param))} are invalid (check number/dimension/PSD).")

    return {
        "input_dict": clean_dict,
        "msg": messages
    }




def check_pars_list(volume_model, n_bin=None):
    converged_list = volume_model.get("converged", {})
    par_list = volume_model.get("par", {})
    init_list = volume_model.get("init", {})

    all_par_list = ["a_eta", "a_mu", "var_eta", "var_mu", "r", "x0", "V0"]
    scalar_par_list = ["a_eta", "a_mu", "var_eta", "var_mu", "r"]
    len_expect = {
        "a_eta": 1, "a_mu": 1, "var_eta": 1,
        "var_mu": 1, "r": 1, "x0": 2, "V0": 3
    }

    if n_bin is not None:
        all_par_list.append("phi")
        len_expect["phi"] = int(n_bin)

    unfixed = [k for k, v in converged_list.items() if not v and k in all_par_list]
    fixed = [k for k, v in converged_list.items() if v and k in all_par_list]

    msg = []

    for name in fixed:
        val = par_list.get(name)
        if not isinstance(val, (list, np.ndarray)) or val is None:
            msg.append(f"{name} must be numeric, have no NAs, and no Infs.\n")
            continue
        val = np.asarray(val)
        if np.any(np.isnan(val)) or np.any(np.isinf(val)):
            msg.append(f"{name} must be numeric, have no NAs, and no Infs.\n")
        elif name in scalar_par_list and len(val) != len_expect[name]:
            msg.append(f"Length of volume_model['par']['{name}'] is wrong.\n")

    for name in unfixed:
        val = par_list.get(name)
        if val is not None and not all(np.isnan(val) if isinstance(val, (list, np.ndarray)) else [False]):
            msg.append(f"volume_model['par']['{name}'] and volume_model['converged']['{name}'] are conflicted.\n")

    for name in fixed:
        if name in init_list:
            msg.append(f"{name} is fixed. No need for init.\n")

    unfixed_init = [k for k in unfixed if k in init_list]
    for name in unfixed_init:
        val = init_list.get(name)
        if not isinstance(val, (list, np.ndarray)) or val is None:
            msg.append(f"{name} must be numeric, have no NAs, and no Infs.\n")
            continue
        val = np.asarray(val)
        if np.any(np.isnan(val)) or np.any(np.isinf(val)):
            msg.append(f"{name} must be numeric, have no NAs, and no Infs.\n")
        elif name in scalar_par_list and len(val) != len_expect[name]:
            msg.append(f"Length of volume_model['init']['{name}'] is wrong.\n")

    return msg



import numpy as np

def check_pars_list(volume_model, n_bin=None):
    all_par_list = ["a_eta", "a_mu", "var_eta", "var_mu", "r", "x0", "V0"]
    scalar_par_list = ["a_eta", "a_mu", "var_eta", "var_mu", "r"]
    len_expect = {"a_eta": 1, "a_mu": 1, "var_eta": 1, "var_mu": 1, "r": 1, "x0": 2, "V0": 3}

    if n_bin is not None:
        all_par_list.append("phi")
        len_expect["phi"] = int(n_bin)

    converged_list = volume_model.get("converged", {})
    unfixed = [name for name, value in converged_list.items() if not value and name in all_par_list]
    fixed = [name for name, value in converged_list.items() if value and name in all_par_list]

    msg = []

    for name in fixed:
        if not isinstance(volume_model['par'][name], np.ndarray) or np.any(np.isnan(volume_model['par'][name])) or np.any(np.isinf(volume_model['par'][name])):
            msg.append(f"{name} must be numeric, have no NAs, and no Infs.\n")
        if name in scalar_par_list and len(volume_model['par'][name]) != len_expect[name]:
            msg.append(f"Length of volume_model$par${name} is wrong.\n")

    for name in unfixed:
        if not np.all(np.isnan(volume_model['par'][name])):
            msg.append(f"volume_model$par${name} and volume_model$converged${name} are conflicted.\n")

    for name in fixed:
        if name in volume_model.get("init", {}):
            msg.append(f"{name} is fixed. No need for init.\n")

    unfixed_init = [name for name in unfixed if name in volume_model.get("init", {})]
    for name in unfixed_init:
        if not isinstance(volume_model['init'][name], np.ndarray) or np.any(np.isnan(volume_model['init'][name])) or np.any(np.isinf(volume_model['init'][name])):
            msg.append(f"{name} must be numeric, have no NAs, and no Infs.\n")
        if name in scalar_par_list and len(volume_model['init'][name]) != len_expect[name]:
            msg.append(f"Length of volume_model$init${name} is wrong.\n")

    return msg


def is_volume_model(volume_model, n_bin=None):
    # Check for required components
    el = ["converged", "par", "init"]
    missing_elements = [e for e in el if e not in volume_model]
    if missing_elements:
        raise ValueError(f"Elements {', '.join(missing_elements)} are missing from the model.\n")

    msg = []
    all_pars_name = ["a_eta", "a_mu", "var_eta", "var_mu", "r", "phi", "x0", "V0"]
    missing_pars = [p for p in all_pars_name if p not in volume_model['par']]
    if missing_pars:
        msg.append(f"Elements {', '.join(missing_pars)} are missing from volume_model$par.\n")

    missing_converged = [p for p in all_pars_name if p not in volume_model['converged']]
    if missing_converged:
        msg.append(f"Elements {', '.join(missing_converged)} are missing from volume_model$converged.\n")

    if msg:
        raise ValueError("".join(msg))

    # Check converged
    invalid_converged = [f for f in volume_model['converged'].values() if not isinstance(f, bool)]
    if invalid_converged:
        raise ValueError("Elements in volume_model$converged must be TRUE/FALSE.\n")

    # Check no NA inf and dimension
    msg = check_pars_list(volume_model, n_bin)
    if msg:
        raise ValueError("".join(msg))


def fetch_par_log(par_log, index):
    return np.column_stack([log[index] for log in par_log])


def calculate_mape(referenced_data, predicted_data):
    referenced_data = np.array(referenced_data)
    predicted_data = np.array(predicted_data)
    return np.mean(np.abs(predicted_data - referenced_data) / referenced_data)


def calculate_mae(referenced_data, predicted_data):
    referenced_data = np.array(referenced_data)
    predicted_data = np.array(predicted_data)
    return np.mean(np.abs(predicted_data - referenced_data))


def calculate_rmse(referenced_data, predicted_data):
    referenced_data = np.array(referenced_data)
    predicted_data = np.array(predicted_data)
    return np.sqrt(np.mean((predicted_data - referenced_data) ** 2))

