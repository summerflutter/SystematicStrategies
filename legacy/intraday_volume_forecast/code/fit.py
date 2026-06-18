import numpy as np
import pandas as pd
import warnings

def fit_volume(data, fixed_pars=None, init_pars=None, verbose=0, control=None):
    """
    Fit the univariate state-space (UNISS) model to the log-volume data.
    data: pandas DataFrame or numpy ndarray.
    """
    # 1. Error control and clean data
    if not (isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray)):
        raise ValueError("data must be a pandas DataFrame or numpy ndarray.")
    data_mat = clean_data(data)

    # 2. Define volume_model and validate
    volume_model = spec_volume_model(fixed_pars, init_pars)
    is_volume_model(volume_model, data_mat.shape[0])

    # 3. Check if all parameters are fixed
    if sum(volume_model['converged'].values()) == 8:
        if verbose > 0:
            print("All parameters have already been fixed.")
        return volume_model

    # 4. Build control_final
    control_final = {'acceleration': True, 'maxit': 3000, 'abstol': 1e-4, 'log_switch': True}
    if isinstance(control, dict):
        for key in control_final:
            if key in control:
                control_final[key] = control[key]

    # 5. Specify UNISS object
    uniss_obj = specify_uniss(np.log(data_mat), volume_model)

    # 6. Run EM algorithm
    if not control_final['acceleration']:
        em_result = uniss_em_alg(uniss_obj, verbose, control_final)
    else:
        em_result = uniss_em_alg_acc(uniss_obj, verbose, control_final)

    # 7. Update volume_model with EM results
    volume_model['par_log'] = em_result['par_log']
    if em_result['warning_msg']:
        warnings.warn("".join(em_result['warning_msg']))

    volume_model['par'] = em_result['uniss_obj']['par']
    if em_result['convergence']:
        volume_model['converged'] = {k: True for k in volume_model['converged']}
        volume_model['init'] = {}

    # 8. Optional verbose output
    if verbose >= 2:
        print("--- obtained parameters ---")
        par_vis = {k: np.array(v) for k, v in volume_model['par'].items()}
        V0 = par_vis['V0']
        par_vis['V0'] = np.array([[V0[0], V0[1]], [V0[1], V0[2]]])
        for k, v in par_vis.items():
            print(f"{k}: {v}")
        print("---------------------------")

    return volume_model


def uniss_em_alg(uniss_obj, verbose, control):
    """
    Standard EM loop for UNISS model.
    """
    convergence = False
    par_log = [uniss_obj['par'].copy()]

    for i in range(1, control['maxit'] + 1):
        # one EM update
        new_par = uniss_kalman(uniss_obj, mode="em_update")['new_par']

        # log
        if control['log_switch']:
            par_log.append(new_par.copy())

        # compute diff
        old_flat = np.concatenate([np.atleast_1d(uniss_obj['par'][k]) for k in new_par])
        new_flat = np.concatenate([np.atleast_1d(new_par[k]) for k in new_par])
        diff = np.linalg.norm(old_flat - new_flat)

        if verbose >= 1 and i % 25 == 0:
            print(f"iter: {i} diff: {diff:.6e}")

        if diff < control['abstol']:
            convergence = True
            break

        uniss_obj['par'] = new_par.copy()

    warning_msg = []
    if not convergence:
        warning_msg.append(f"Warning! Reached maxit ({control['maxit']}) before convergence.\n")
    elif verbose > 0:
        print(f"Success! abstol test passed at {i} iterations.")

    return {
        'uniss_obj': uniss_obj,
        'convergence': convergence,
        'par_log': par_log,
        'warning_msg': warning_msg
    }


def uniss_em_alg_acc(uniss_obj, verbose, control):
    """
    Accelerated EM loop for UNISS model.
    """
    convergence = False
    par_log = [uniss_obj['par'].copy()]

    for i in range(1, control['maxit'] + 1):
        curr_par = uniss_obj['par'].copy()

        # two successive updates
        new1 = uniss_kalman(uniss_obj, mode="em_update")['new_par']
        uniss_obj['par'] = new1.copy()
        new2 = uniss_kalman(uniss_obj, mode="em_update")['new_par']

        # acceleration step
        new_par = curr_par.copy()

        # accelerate phi
        if not uniss_obj['converged']['phi']:
            r = new1['phi'] - curr_par['phi']
            v = new2['phi'] - new1['phi'] - r
            r_norm = np.linalg.norm(r)
            v_norm = np.linalg.norm(v)
            step = -r_norm / v_norm if v_norm != 0 else 0
            new_par['phi'] = curr_par['phi'] - 2*step*r + (step**2)*v

        # accelerate other parameters
        for name in curr_par:
            if name != 'phi':
                arr_curr = np.atleast_1d(curr_par[name])
                arr1 = np.atleast_1d(new1[name])
                arr2 = np.atleast_1d(new2[name])
                r = arr1 - arr_curr
                v = arr2 - arr1 - r
                step = -np.abs(r) / np.where(np.abs(v) > 1e-8, np.abs(v), np.inf)
                accel = np.where(np.abs(v) > 1e-8, arr_curr - step * r, arr2)
                new_par[name] = accel

        # ensure positivity
        for key in ['r', 'var_eta', 'var_mu']:
            if new_par[key] < 0:
                new_par[key] = new2[key]

        # log
        if control['log_switch']:
            par_log.append(new_par.copy())

        # stopping criterion
        rvec = np.concatenate([np.atleast_1d(new1[k]) for k in new1])
        vvec = np.concatenate([np.atleast_1d(new2[k]) for k in new2])
        diff = np.linalg.norm(rvec - vvec)

        if verbose >= 1 and i % 5 == 0:
            print(f"iter: {i} diff: {diff:.6e}")

        if diff < control['abstol']:
            convergence = True
            break

        uniss_obj['par'] = new_par.copy()

    warning_msg = []
    if not convergence:
        warning_msg.append(f"Warning! Reached maxit ({control['maxit']}) before convergence.\n")
    elif verbose > 0:
        print(f"Success! abstol test passed at {i} iterations.")

    return {
        'uniss_obj': uniss_obj,
        'convergence': convergence,
        'par_log': par_log,
        'warning_msg': warning_msg
    }
