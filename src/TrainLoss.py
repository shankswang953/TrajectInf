
from os import sync
from unittest import result
from numpy import divide
import torch
from typing import Tuple, List, Dict
from src.Loss import calculate_density_loss, compute_kl_divergence, compute_pdf_lossNegLog, compute_pdf_lossMSE, get_geodesic_loss
from src.DataLoad import Sampling
from torchdiffeq import odeint
from src.DataLoad import create_mixture_gaussian
from src.Loss import Manifold_reconstruction_loss, MMD_loss, relative_log_pdf
from src.MapSpace import map_to_nearest_manifold, map_whole_trajectory2another_manifold, knn_aux_mean_distance_loss, smooth_trajectory, project_trajectory
from geomloss import SamplesLoss
from src.DataLoad import torch_record
from torch.utils.checkpoint import checkpoint


def short_term_loss(
    func,
    t_idx: int,
    time_points: List[float],
    primary_data: List[torch.Tensor],
    secondary_data: List[torch.Tensor] = None,
    manifold_mixture: torch.distributions.MixtureSameFamily = None,
    secondary_mixture: torch.distributions.MixtureSameFamily = None,
    time_steps_list: List[Tuple] = None,
    density_precompute_list: List[Tuple] = None,
    device: torch.device = None,
    args = None,
    writer=None,
    itr=None,
    rare_files=None
) -> Dict[str, torch.Tensor]:
    """
    Compute forward and backward losses for a single time step
    
    Parameters:
    - func: Neural ODE function
    - t_idx: Current time index
    - time_points: List of time points
    - primary_data: List of primary space data
    - secondary_data: List of secondary space data
    - manifold_mixture: Mixture of manifolds
    - time_steps_list: List of time steps
    - density_precompute_list: List of precomputed density parameters
    - device: Computation device
    - args: Arguments object
    - writer: TensorBoard SummaryWriter (optional)
    - itr: Current iteration number (optional)
    
    Returns:
    - Dictionary containing various losses
    """
    # get arguments
    forward_density_loss_flag = args.forward_density_loss
    backward_density_loss_flag = args.backward_density_loss
    pdf_loss_flagMSE = args.pdf_lossMSE
    pdf_loss_flagNegLog = args.pdf_lossNegLog
    kl_loss_flag = args.kl_loss
    geo_loss_flag = args.geo_loss
    manifold_reconstruction_loss_flag = args.manifold_reconstruction_loss
    mmd_criterion = MMD_loss()
    w2_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.005, scaling=0.5).to(device)
    # Get current and next time point data
    z_curr_primal = primary_data[t_idx]
    z_next_primal = primary_data[t_idx + 1]
    z_support = primary_data[len(primary_data) - 1]
    
    # Calculate mass ratio
    curr_mass_ratio = z_next_primal.shape[0] / z_curr_primal.shape[0]
    curr_mass_ratio = torch.tensor(curr_mass_ratio, dtype=torch.float32, device=device)
    
    
    # Create combined manifold
    manifold_concat = torch.cat((z_curr_primal, z_next_primal, z_support), dim=0)
    
    # Get time steps
    time_steps_backward, time_steps_forward, _, _ = time_steps_list[t_idx]
    
    # Get precomputed density parameters
    _, _, _, current_mixture = density_precompute_list[t_idx]
    _, _, _, next_mixture = density_precompute_list[t_idx + 1]
    
    # Initialize results dictionary
    results = {
        'backward_losses': {},
        'forward_losses': {},
        'sync_losses': {},
        'total_loss': 0.0
    }
    
    # ============ BACKWARD EVOLUTION LOSS CALCULATION ============
    if forward_density_loss_flag or backward_density_loss_flag or pdf_loss_flagMSE or pdf_loss_flagNegLog or kl_loss_flag or geo_loss_flag:
        # Sample terminal points
        
        z_next_primal_sample, logp_diff_next = Sampling(args.num_samples, device, z_next_primal, rare_files=rare_files)
        

        
        if args.UnbalancedOT:
            g_primal = torch.zeros(args.num_samples, 1).to(device)
            # Backward time evolution
            zt_primal_sample, logp_diff_t, _ = odeint(
                func,
                (z_next_primal_sample, logp_diff_next, g_primal),
                time_steps_backward,
                atol=1e-4,
                rtol=1e-4,
                method='rk4',
            )
        else:
            # Backward time evolution
            zt_primal_sample, logp_diff_t = odeint(
                func,
                (z_next_primal_sample, logp_diff_next),
                time_steps_backward,
                atol=1e-4,
                rtol=1e-4,
                method='rk4',
            )

        
        z_curr_primal_sample = zt_primal_sample[-1]
        logp_diff_curr = logp_diff_t[-1]
        

        
        # Calculate density loss
        if backward_density_loss_flag:
            density_loss = 0
            for i in range(len(zt_primal_sample) - 1):
                density_loss = density_loss + calculate_density_loss(
                    zt_primal_sample[i], 
                    manifold_concat, 
                    k=args.density_k, 
                    hinge_value=args.density_hinge
                )
            density_loss = density_loss + calculate_density_loss(
                zt_primal_sample[-1], 
                z_curr_primal, 
                k=args.exact_density_k, 
                hinge_value=args.exact_density_hinge
            ) * args.terminal_density_weight
            density_loss = density_loss * args.density_coefficient
            results['backward_losses']['density_loss'] = density_loss
            

    
        if args.manifold_logdensity_loss:
            manifold_logdensity_loss = 0
            manifold_pdf_loss = 0
            for i in range(len(zt_primal_sample) - 1):
                log_current_predicted = manifold_mixture.log_prob(zt_primal_sample[i])
                manifold_pdf_loss = manifold_pdf_loss - (log_current_predicted - logp_diff_t[i]).mean() 
                
            manifold_logdensity_loss = manifold_pdf_loss * args.density_coefficient
            results['backward_losses']['manifold_logdensity_loss'] = manifold_logdensity_loss
            
           
        print("after manifold logdensity loss")
        torch_record()

            
        # Calculate KL divergence loss
        current_kl = 0
        if kl_loss_flag:
            current_kl = compute_kl_divergence(z_curr_primal, z_curr_primal_sample) * args.kl_coefficient
            results['backward_losses']['kl_loss'] = current_kl
            
        # Calculate MMD loss
        mmd_loss = 0
        if args.mmd_loss:
            current_mmd = mmd_criterion(z_curr_primal_sample, z_curr_primal)
            mmd_loss = current_mmd * args.kl_coefficient
            results['backward_losses']['mmd_loss'] = mmd_loss
        
        # Calculate geodesic loss
        geo_loss = 0
        if geo_loss_flag:
            geo_loss = get_geodesic_loss(z_curr_primal_sample, z_curr_primal) * args.geo_coefficient
            results['backward_losses']['geo_loss'] = geo_loss
        
        # Calculate PDF loss
        loss_pdf = 0
        if pdf_loss_flagMSE:
            loss_pdf = compute_pdf_lossMSE(
                z_next_primal_sample, # initial time query points
                z_curr_primal_sample, # terminal time query points
                logp_diff_next, # log of Jacobian determinant for initial time
                logp_diff_curr, # log of Jacobian determinant for terminal time
                next_mixture, # initial time GMM parameters
                current_mixture, # terminal time GMM parameters
                curr_mass_ratio # mass ratio between initial and terminal time
            ) * args.pdf_coefficient
            results['backward_losses']['pdf_loss'] = loss_pdf

        
        if pdf_loss_flagNegLog:
            loss_pdf = compute_pdf_lossNegLog(
                z_next_primal_sample, # initial time query points
                z_curr_primal_sample, # terminal time query points
                logp_diff_next, # log of Jacobian determinant for initial time
                logp_diff_curr, # log of Jacobian determinant for terminal time
                next_mixture, # initial time GMM parameters
                current_mixture, # terminal time GMM parameters
                curr_mass_ratio # mass ratio between initial and terminal time
            ) * args.pdf_coefficient
            results['backward_losses']['pdf_loss'] = loss_pdf
        print("after pdf loss")
        torch_record()
        
        # Calculate manifold reconstruction loss
        manifold_reconstruction_loss = 0
        if manifold_reconstruction_loss_flag:
            manifold_reconstruction_loss = Manifold_reconstruction_loss(
                zt_primal_sample,
                manifold_concat,
                device,
            ) * args.manifold_reconstruction_loss_coefficient
        
        # Total backward loss
        total_backward_loss = 0
        if backward_density_loss_flag:
            total_backward_loss += density_loss
        if args.mmd_loss:
            total_backward_loss += mmd_loss
        if kl_loss_flag:
            total_backward_loss += current_kl
        if geo_loss_flag:
            total_backward_loss += geo_loss
        if pdf_loss_flagMSE:
            total_backward_loss += loss_pdf
        if pdf_loss_flagNegLog:
            total_backward_loss += loss_pdf
        if manifold_reconstruction_loss_flag:
            total_backward_loss += manifold_reconstruction_loss
        if args.manifold_logdensity_loss:
            total_backward_loss += manifold_logdensity_loss
        
        results['backward_losses']['total'] = total_backward_loss
        results['total_loss'] += total_backward_loss
        
        # Log losses if writer is provided
        if writer is not None and itr is not None:
            if pdf_loss_flagMSE:
                writer.add_scalar('ShortTermLoss/backward_pdf', loss_pdf.mean()/args.pdf_coefficient, itr)
            if pdf_loss_flagNegLog:
                writer.add_scalar('ShortTermLoss/backward_pdf', loss_pdf.mean()/args.pdf_coefficient, itr)
            if backward_density_loss_flag:
                writer.add_scalar('ShortTermLoss/backward_density', density_loss.mean()/args.density_coefficient, itr)
            if args.mmd_loss:
                writer.add_scalar('ShortTermLoss/backward_mmd', mmd_loss.mean()/args.kl_coefficient, itr)
            if kl_loss_flag:
                writer.add_scalar('ShortTermLoss/backward_kl', current_kl.mean()/args.kl_coefficient, itr)
            if geo_loss_flag:
                writer.add_scalar('ShortTermLoss/backward_geo', geo_loss.mean()/args.geo_coefficient, itr)
            if manifold_reconstruction_loss_flag:
                writer.add_scalar('ShortTermLoss/backward_manifold_reconstruction', manifold_reconstruction_loss.mean()/args.manifold_reconstruction_loss_coefficient, itr)
            if args.manifold_logdensity_loss:
                writer.add_scalar('ShortTermLoss/backward_manifold_logdensity', manifold_logdensity_loss.mean()/args.density_coefficient, itr)
                

                
    # ============ FORWARD EVOLUTION LOSS CALCULATION ============
    # Sample initial points
    z_current_primal_forward_sample, logp_diff_current_forward = Sampling(args.num_samples, device, z_curr_primal)
    g_curr = torch.zeros(args.num_samples, 1).to(device)
    
    # Forward time evolution
    if args.UnbalancedOT:
        zt_primal_forward_sample, logp_diff_t_forward, g_cum_t_forward = odeint(
            func,
            (z_current_primal_forward_sample, logp_diff_current_forward, g_curr),
            time_steps_forward,
            atol=1e-4,
            rtol=1e-4,
            method='rk4',
        )
    else:
        zt_primal_forward_sample, logp_diff_t_forward = odeint(
            func,
            (z_current_primal_forward_sample, logp_diff_current_forward),
            time_steps_forward,
            atol=1e-4,
            rtol=1e-4,
            method='rk4',
        )
    

    
    # Calculate forward density loss
    forward_density_loss = 0
    if forward_density_loss_flag:
        for i in range(len(zt_primal_forward_sample) - 1):
            forward_density_loss = forward_density_loss + calculate_density_loss(
                zt_primal_forward_sample[i], 
                manifold_concat, 
                k=args.density_k, 
                hinge_value=args.density_hinge
            )
        # Terminal forward density loss
        forward_density_loss = forward_density_loss + calculate_density_loss(
            zt_primal_forward_sample[-1], 
            z_next_primal, 
            k=args.exact_density_k, 
            hinge_value=args.exact_density_hinge
        ) * args.terminal_density_weight
        
        forward_density_loss = forward_density_loss * args.density_coefficient
        results['forward_losses']['density_loss'] = forward_density_loss
    
    # Calculate terminal forward density loss (for evaluation)
    section_forward_density_loss = calculate_density_loss(
        zt_primal_forward_sample[-1], 
        z_next_primal, 
        k=args.exact_density_k, 
        hinge_value=args.exact_density_hinge
    )
    results['forward_losses']['terminal_density'] = section_forward_density_loss
    

    
    dz_dt_list = []
    g_t_list = []
    
    
    if args.UnbalancedOT:
        for t_i_forward, zi_primal_forward, logp_diff_i_forward, g_i_forward in zip(
            time_steps_forward, 
            zt_primal_forward_sample, 
            logp_diff_t_forward, 
            g_cum_t_forward
        ):
            zi_primal_forward.requires_grad_(True)
            dz_dt_i_forward, _, g_i_forward = func(t_i_forward, (zi_primal_forward, logp_diff_i_forward, g_i_forward))
            dz_dt_list.append(dz_dt_i_forward)
            g_t_list.append(g_i_forward)
        g_t_all = torch.stack(g_t_list, dim=0)
        g_norm_squared = (g_t_all ** 2).squeeze(-1)
        g_cum_t_forward = g_cum_t_forward.squeeze(-1)
        
    else:
        for t_i_forward, zi_primal_forward, logp_diff_i_forward in zip(
            time_steps_forward, 
            zt_primal_forward_sample, 
            logp_diff_t_forward
        ):
            zi_primal_forward.requires_grad_(True)
            dz_dt_i_forward, _ = func(t_i_forward, (zi_primal_forward, logp_diff_i_forward))
            dz_dt_list.append(dz_dt_i_forward)
        
        

    
    dz_dt_all = torch.stack(dz_dt_list, dim=0)
    dz_dt_norm_squared = (dz_dt_all ** 2).sum(dim=2)


        
    
    
    
    # Calculate forward KL divergence
    kl_divergence_forward = 0
    if kl_loss_flag:
        kl_divergence_forward = compute_kl_divergence(
            zt_primal_forward_sample[-1], 
            z_next_primal
        ) * args.kl_coefficient
        results['forward_losses']['kl_loss'] = kl_divergence_forward
        
    # Calculate geodesic loss
    forward_geo_loss = 0
    if args.forward_geo_loss:
        forward_geo_loss = get_geodesic_loss(zt_primal_forward_sample[-1], z_next_primal) * args.geo_coefficient
        results['forward_losses']['geo_loss'] = forward_geo_loss
    
    # Calculate energy components
    if args.UnbalancedOT:
        energy_components = (args.dt * (dz_dt_norm_squared) * torch.exp(g_cum_t_forward))
    else:
        energy_components = (args.dt * (dz_dt_norm_squared ** 2)  )
    energy_loss = energy_components.mean() * args.forward_energy_coefficient
    results['forward_losses']['energy_loss'] = energy_loss
    results['total_loss'] += energy_loss
    
    
     # ============ SYNCHRONIZATION LOSS CALCULATION ============
     # backward evolution
    if args.sync_loss and secondary_data is not None:
        sync_loss = 0
        
        
        # project the forward data to the secondary data
        whole_primal = torch.cat(primary_data, dim=0)
        whole_secondary = torch.cat(secondary_data, dim=0)
        mapped_mainfold_list = []
        
        time_length = zt_primal_sample.shape[0]
        

        mapped_indices, weights, path_indices = map_whole_trajectory2another_manifold(
            traj_points=zt_primal_sample,  # (N, Sample, dim)
            support_points_main=whole_primal,
            support_points_aux=whole_secondary,
            k=20,
            type=args.map_type
        )
        for i in range(time_length):
            if i == 0:
                mapped_secondary = whole_secondary[path_indices[i]]
                mapped_mainfold_list.append(mapped_secondary)
            else:
                mapped_secondary = whole_secondary[mapped_indices[i]]
                mapped_mainfold = torch.sum(mapped_secondary * weights[i].unsqueeze(-1), dim=1)
                mapped_mainfold_list.append(mapped_mainfold)

        secondary_trajectory = torch.stack(mapped_mainfold_list, dim=0)  # (N, Sample, dim)
        smoothed_secondary_trajectory = smooth_trajectory(secondary_trajectory, window_size=7)
        
        del mapped_mainfold_list, mapped_indices, weights, path_indices, whole_primal, whole_secondary
        

        # secondary energy loss
        secondary_energy_loss = 0
        

        '''
        if current_mmd < 1:
            current_sec_energy = args.forward_energy_coefficient * 100
        else:
            current_sec_energy = args.forward_energy_coefficient 
        '''


        
        # generate point cloud pairs
        point_cloud_pairs = [
            (secondary_trajectory[i], secondary_trajectory[i+1])
            for i in range(len(secondary_trajectory) - 1)
        ]
        batch_losses = [torch.norm(pc1 - pc2, p=2) for pc1, pc2 in point_cloud_pairs]
        #batch_losses = [w2_loss((pc1), (pc2)) for pc1, pc2 in point_cloud_pairs]
        secondary_energy_loss = torch.mean(torch.stack(batch_losses))
        results['sync_losses']['secondary_energy_loss'] = secondary_energy_loss * args.sync_energy_coefficient
        sync_loss = sync_loss + secondary_energy_loss * args.sync_energy_coefficient
        
        
        print("sync point cloud pairs")
        torch_record()
        

        curr_secondary = secondary_data[t_idx]
        next_secondary = secondary_data[t_idx+1]
        if args.support_points is not None:
            support_points_secondary = secondary_data[-1]
            secondary_manifold_concat = torch.cat([curr_secondary, next_secondary, support_points_secondary], dim=0)
        else:
            secondary_manifold_concat = torch.cat([curr_secondary, next_secondary], dim=0)

        
        # secondary manifold loss
        if args.sync_manifold_loss:
            secondary_manifold_loss_list = []
            for i in range(len(smoothed_secondary_trajectory) - 1):
                secondary_manifold_loss = calculate_density_loss(
                    smoothed_secondary_trajectory[i], 
                    secondary_manifold_concat, 
                    k=args.density_k, 
                    hinge_value=args.density_hinge
                )
                secondary_manifold_loss_list.append(secondary_manifold_loss * args.density_coefficient)
            secondary_manifold_loss = torch.stack(secondary_manifold_loss_list)
            secondary_manifold_loss = secondary_manifold_loss.mean()
            


            
            results['sync_losses']['secondary_manifold_loss'] = secondary_manifold_loss 
            sync_loss = sync_loss + secondary_manifold_loss 
        
        
        # sync mmd loss
        if args.sync_mmd_loss:
            sync_mmd_loss = mmd_criterion(secondary_trajectory[-1], curr_secondary)
            results['sync_losses']['sync_mmd_loss'] = sync_mmd_loss * args.sync_mmd_coefficient
            sync_loss = sync_loss + sync_mmd_loss * args.sync_mmd_coefficient
        if args.sync_kl_divergence_loss:
            sync_kl_divergence_loss = compute_kl_divergence(secondary_trajectory[-1], curr_secondary)
            results['sync_losses']['sync_kl_divergence_loss'] = sync_kl_divergence_loss * args.sync_mmd_coefficient
            sync_loss = sync_loss + sync_kl_divergence_loss * args.sync_mmd_coefficient
            
        print("before sync pdf loss")
        torch_record()
        # sync pdf loss
        if args.sync_pdf_loss:
            sync_manifold_pdf_loss = 0
            for i in range(len(smoothed_secondary_trajectory) - 1):
                z_secondary_sample = smoothed_secondary_trajectory[i]

                sec_pdf = - secondary_mixture.log_prob(z_secondary_sample)
                sync_manifold_pdf_loss = sync_manifold_pdf_loss + sec_pdf.mean()
                
            results['sync_losses']['sync_manifold_logdensity_loss'] = sync_manifold_pdf_loss
            sync_loss = sync_loss + sync_manifold_pdf_loss * args.sync_pdf_coefficient
            
        
        print("after sync pdf loss")
        torch_record()
        
    total_forward_loss = 0
    
    # Calculate total forward loss
    if kl_loss_flag:
        total_forward_loss += kl_divergence_forward
    if forward_density_loss_flag:
        total_forward_loss += forward_density_loss
    if args.forward_geo_loss:
        total_forward_loss += forward_geo_loss
        
    results['forward_losses']['total'] = total_forward_loss
    results['total_loss'] += total_forward_loss
    
    # sync loss
    if args.sync_loss and secondary_data is not None:
        total_sync_loss = sync_loss * args.sync_coefficient
    
        results['sync_losses']['total'] = total_sync_loss
        results['total_loss'] +=  total_sync_loss
    
    # Log forward losses if writer is provided
    if writer is not None and itr is not None:

        if forward_density_loss_flag:
            writer.add_scalar('Loss/Forward_density', section_forward_density_loss.mean(), itr)
        if args.forward_geo_loss:
            writer.add_scalar('Loss/Forward_geo', forward_geo_loss.mean()/args.geo_coefficient, itr)
        writer.add_scalar('Loss/Forward_energy', energy_loss/args.forward_energy_coefficient, itr)
        if args.sync_loss and secondary_data is not None:
            writer.add_scalar('Loss/Sync_energy', secondary_energy_loss, itr)
            if args.sync_manifold_loss:
                writer.add_scalar('Loss/Sync_manifold', secondary_manifold_loss, itr)
            #writer.add_scalar('Loss/Sync_consistency', consistency_loss, itr)
            if args.sync_pdf_loss:
                writer.add_scalar('Loss/Sync_pdf', sync_manifold_pdf_loss, itr)

            if args.sync_mmd_loss:
                writer.add_scalar('Loss/Sync_mmd', sync_mmd_loss, itr)
            if args.sync_kl_divergence_loss:
                writer.add_scalar('Loss/Sync_kl_divergence', sync_kl_divergence_loss, itr)
    
    
    
    return results






def long_term_loss(
    func,
    t_idx,  
    primary_data,
    time_points,
    time_steps_list,
    density_precompute_list,
    total_primal,  
    device,
    args,  
    writer=None,
    itr=None
) -> Dict[str, torch.Tensor]:
    """
    Compute long-term losses from initial time point (t=0) to a specific time point (t_idx)
    
    Parameters:
    - func: Neural ODE function
    - t_idx: Target time index to compute losses for (not including 0)
    - primary_data: List of primary space data at different time points
    - time_points: List of time points
    - time_steps_list: List of time steps for ODE integration
    - density_precompute_list: List of precomputed density parameters
    - total_primal: Concatenated data from all time points
    - device: Computation device
    - loss_params: Dictionary of loss parameters
    - writer: TensorBoard SummaryWriter (optional)
    - itr: Current iteration number (optional)
    
    Returns:
    - Dictionary containing backward and forward losses
    """
    # Extract parameters from dictionary with defaults
    forward_density_loss_flag = args.forward_density_loss
    backward_density_loss_flag = args.backward_density_loss
    pdf_loss_flagMSE = args.pdf_lossMSE
    pdf_loss_flagNegLog = args.pdf_lossNegLog
    kl_loss_flag = args.kl_loss
    

    # Always start from initial time point (t=0)
    z_curr_primal = primary_data[0]
    z_next_primal = primary_data[t_idx]
    
    # Calculate mass ratio
    curr_mass_ratio = z_next_primal.shape[0] / z_curr_primal.shape[0]
    curr_mass_ratio = torch.tensor(curr_mass_ratio, dtype=torch.float32, device=device)
    
    # Get long-term time steps
    _, _, time_long_backward, time_long_forward = time_steps_list[t_idx]  
    
    # Get precomputed density parameters
    current_weights, current_means, current_precisions, current_mixture = density_precompute_list[0]
    next_weights, next_means, next_precisions, next_mixture = density_precompute_list[t_idx]
    
    # Initialize results dictionary
    results = {
        'backward_losses': {},
        'forward_losses': {},
        'total_loss': 0.0
    }
    
    # ============ BACKWARD EVOLUTION ============
    # Sample terminal points
    z_next_primal_sample, logp_diff_next = Sampling(args.num_samples, device, z_next_primal)
    g_primal_long = torch.zeros(args.num_samples, 1).to(device)
    
    # Backward time evolution
    zt_primal_sample, logp_diff_t, g_t_long = odeint(
        func,
        (z_next_primal_sample, logp_diff_next, g_primal_long),
        time_long_backward,
        atol=1e-4,
        rtol=1e-4,
        method='rk4',
    )
    
    z_curr_primal_sample, logp_diff_curr, g_curr_long = zt_primal_sample[-1], logp_diff_t[-1], g_t_long[-1]
    time_step = len(zt_primal_sample)
    
    # Calculate backward density loss
    if backward_density_loss_flag:
        backward_density_loss = 0
        for i in range(time_step-1):
            backward_density_loss = backward_density_loss + calculate_density_loss(
                zt_primal_sample[i], 
                total_primal, 
                k=args.density_k, 
                hinge_value=args.density_hinge
            )
        backward_density_loss = backward_density_loss + calculate_density_loss(
            zt_primal_sample[-1], 
            z_curr_primal, 
            k=args.exact_density_k, 
            hinge_value=args.exact_density_hinge
        ) * args.terminal_density_weight
        backward_density_loss = backward_density_loss * args.density_coefficient
        results['backward_losses']['density_loss'] = backward_density_loss
    
    if kl_loss_flag:
        # Calculate KL divergence loss
        current_kl = compute_kl_divergence(z_curr_primal_sample, z_curr_primal) * args.kl_coefficient
        results['backward_losses']['kl_loss'] = current_kl
    
    if pdf_loss_flagMSE:
        # Calculate PDF loss
        long_term_loss_pdf = compute_pdf_lossMSE(
            z_next_primal_sample, #z_initial_primal_sample
            z_curr_primal_sample, #z_terminal_primal_sample
            logp_diff_next, #logp_diff_initial
            logp_diff_curr, #logp_diff_terminal
            next_mixture, #initial_mixture
            current_mixture, #terminal_mixture
            curr_mass_ratio #mass_ratio
            ) * args.pdf_coefficient
        results['backward_losses']['pdf_loss'] = long_term_loss_pdf
    
    if pdf_loss_flagNegLog:
        # Calculate PDF loss
        long_term_loss_pdf = compute_pdf_lossNegLog(
            z_next_primal_sample, #z_initial_primal_sample
            z_curr_primal_sample, #z_terminal_primal_sample
            logp_diff_next, #logp_diff_initial
            logp_diff_curr, #logp_diff_terminal
            next_mixture, #initial_mixture
            current_mixture, #terminal_mixture
            curr_mass_ratio #mass_ratio
        ) * args.pdf_coefficient
        results['backward_losses']['pdf_loss'] = long_term_loss_pdf
    # Total backward loss
    total_backward_loss = 0; #backward_density_loss + long_term_loss_pdf + current_kl
    if backward_density_loss_flag:
        total_backward_loss += backward_density_loss
    if kl_loss_flag:
        total_backward_loss += current_kl
    if pdf_loss_flagMSE:
        total_backward_loss += long_term_loss_pdf
    if pdf_loss_flagNegLog:
        total_backward_loss += long_term_loss_pdf
        
    results['backward_losses']['total'] = total_backward_loss
    
    if writer is not None and itr is not None:
        if pdf_loss_flagMSE:
            writer.add_scalar('LongTermLoss/Backward_log_pdf', long_term_loss_pdf.mean()/args.pdf_coefficient, itr)
        if pdf_loss_flagNegLog:
            writer.add_scalar('LongTermLoss/Backward_log_pdf', long_term_loss_pdf.mean()/args.pdf_coefficient, itr)
        if backward_density_loss_flag:
            writer.add_scalar('LongTermLoss/Backward_density', backward_density_loss.mean()/args.density_coefficient, itr)
        if kl_loss_flag:
            writer.add_scalar('LongTermLoss/Backward_kl', current_kl.mean()/args.kl_coefficient, itr)
    
    # ============ FORWARD EVOLUTION ============
    # Initial batch for forward evolution
    z_current_primal_forward_sample, logp_diff_current_forward = Sampling(args.num_samples, device, z_curr_primal)
    g_curr_long = torch.zeros(args.num_samples, 1).to(device)
    
    # Forward time evolution
    zt_long_forward_sample, logp_diff_t_forward, g_long_t_forward = odeint(
        func,
        (z_current_primal_forward_sample, logp_diff_current_forward, g_curr_long),
        time_long_forward,
        atol=1e-4,
        rtol=1e-4,
        method='rk4',
    )
    
    
    if args.forward_density_loss:
        # Forward density loss
        forward_density_loss = 0
        time_step = len(zt_long_forward_sample)
        for i in range(time_step-1):
            forward_density_loss = forward_density_loss + calculate_density_loss(
                zt_long_forward_sample[i], 
                total_primal, 
                k=args.density_k, 
                hinge_value=args.density_hinge
            )
        forward_density_loss = forward_density_loss + calculate_density_loss(
            zt_long_forward_sample[-1], 
            z_next_primal, 
            k=args.exact_density_k, 
            hinge_value=args.exact_density_hinge
        ) * args.terminal_density_weight
        forward_density_loss = forward_density_loss * args.density_coefficient
        results['forward_losses']['density_loss'] = forward_density_loss
    
    if writer is not None and itr is not None:
        if forward_density_loss_flag:
            writer.add_scalar('LongTermLoss/Forward_density', forward_density_loss.mean()/args.density_coefficient, itr)
    
    # Energy loss calculation
    dz_dt_list = []
    g_t_list = []
    for t_i_forward, zi_primal_forward, logp_diff_i_forward, g_i_forward in zip(
        time_long_forward, zt_long_forward_sample, logp_diff_t_forward, g_long_t_forward
    ):
        zi_primal_forward.requires_grad_(True)
        dz_dt_i_forward, _, g_i_forward = func(t_i_forward, (zi_primal_forward, logp_diff_i_forward, g_i_forward))
        dz_dt_list.append(dz_dt_i_forward)
        g_t_list.append(g_i_forward)
    
    dz_dt_all = torch.stack(dz_dt_list, dim=0)  
    dz_dt_norm_squared = (dz_dt_all ** 2).sum(dim=2)
    g_t_all = torch.stack(g_t_list, dim=0)
    g_norm_squared = (g_t_all ** 2).squeeze(-1)
    g_long_t_forward = g_long_t_forward.squeeze(-1)
    
    # Energy loss with g-weight
    loss2_components = (args.dt * (dz_dt_norm_squared + args.energy_g_weight * g_norm_squared) 
                         * torch.exp(g_long_t_forward))
    loss2_energy = loss2_components.mean() * args.forward_energy_coefficient
    results['forward_losses']['energy_loss'] = loss2_energy
    
    # KL divergence for forward pass
    if kl_loss_flag:
        kl_divergence_forward = compute_kl_divergence(zt_long_forward_sample[-1], z_next_primal) * args.kl_coefficient
        results['forward_losses']['kl_loss'] = kl_divergence_forward
    
    
    # Terminal forward density loss
    terminal_forward_density_loss = calculate_density_loss(
        zt_long_forward_sample[-1], 
        z_next_primal, 
        k=args.exact_density_k, 
        hinge_value=args.exact_density_hinge
    )
    results['forward_losses']['terminal_density'] = terminal_forward_density_loss
    
    if writer is not None and itr is not None:
        writer.add_scalar(f'LongTermLoss/Forward_density_terminal', 
                         terminal_forward_density_loss, itr)
    
    # Total forward loss
    total_forward_loss = loss2_energy  * args.forward_energy_coefficient # + forward_pdf_loss + kl_divergence_forward + forward_density_loss
    if pdf_loss_flagNegLog:
        total_forward_loss += long_term_loss_pdf
    if forward_density_loss_flag:
        total_forward_loss += forward_density_loss

        
    results['forward_losses']['total'] = total_forward_loss
    
    # Total loss
    results['total_loss'] = total_backward_loss + total_forward_loss
    
    if writer is not None and itr is not None:
        writer.add_scalar('LongTermLoss/Forward_energy', loss2_energy.mean(), itr)
        writer.add_scalar('LongTermLoss/Forward_kl', kl_divergence_forward.mean()/args.kl_coefficient, itr)
        if forward_density_loss_flag:
            writer.add_scalar('LongTermLoss/Forward_density', forward_density_loss.mean()/args.density_coefficient, itr)
    return results



def concatenate_long_term_loss(
    func, 
    primary_data,
    secondary_data,
    time_points,
    time_steps_list,
    density_precompute_list,
    total_primal,  
    total_secondary,
    secondary_mixture,
    support_data,
    device,
    args,  
    writer=None,
    itr=None
) -> Dict[str, torch.Tensor]:
    """
    
    Parameters:
    - func: Neural ODE function
    - primary_data: List of primary space data at different time points
    - time_points: List of time points
    - whole time steps: List of time steps for ODE integration
    - density_precompute_list: List of precomputed density parameters
    - total_primal: Concatenated data from all time points
    - device: Computation device
    - loss_params: Dictionary of loss parameters
    - writer: TensorBoard SummaryWriter (optional)
    - itr: Current iteration number (optional)
    
    Returns:
    - Dictionary containing backward and forward losses
    """
    
    
    Inital_data = primary_data[0]
    if support_data:
        Terminal_data = primary_data[-2]
    else:
        Terminal_data = primary_data[-1]
    
    whole_time_backward = torch.linspace(time_points[-1], time_points[0], int((time_points[-1]-time_points[0])/args.dt)+1).to(device)
    mmd_criterion = MMD_loss()
    w2_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.005, scaling=0.5).to(device)  
    # backward evolution
    z_terminal_sample, _= Sampling(args.num_samples, device, Terminal_data, rare_files=rare_files)
    
    all_samples = [z_terminal_sample]
    
    
    density_loss = 0
    mmd_loss = 0
    pdf_loss = 0
    manifold_reconstruction_loss = 0
    total_backward_loss = 0
    if support_data:
        data_support = primary_data[len(primary_data) - 1]
    
    for t_idx in range(len(time_points) - 1):
        
            
        time_points_backward = torch.linspace(time_points[-t_idx-1], time_points[-t_idx-2], int((time_points[-t_idx-1]-time_points[-t_idx-2])/args.dt)+1).to(device)
        
        start_sample = all_samples[-1]
        start_logp_diff = torch.zeros(torch.tensor(start_sample.shape[0]), 1).to(device)
        start_g = torch.zeros(torch.tensor(start_sample.shape[0]), 1).to(device)
        
        zt_primal_sample, logp_diff_t, g_t_long = odeint(
            func,
            (start_sample, start_logp_diff, start_g),
            time_points_backward,
            atol=1e-4,
            rtol=1e-4,
            method='rk4',
        )
        
        prediction_data = zt_primal_sample[-1]
        prediction_logp_diff = logp_diff_t[-1]
        prediction_g = g_t_long[-1]
        
        if support_data:
            new_sample, _ = Sampling(args.num_samples, device, primary_data[-t_idx-3])
            
        else:
            new_sample, _ = Sampling(args.num_samples, device, primary_data[-t_idx-2])
            
        merge_sample = torch.cat((prediction_data, new_sample), dim=0)
        all_samples.append(merge_sample)


        if support_data:
            z_start_data = primary_data[-t_idx-2]
            z_end_data = primary_data[-t_idx-3]
        else:
            z_start_data = primary_data[-t_idx-1]
            z_end_data = primary_data[-t_idx-2]
        
        _, _, _, start_mixture = density_precompute_list[-t_idx-1]
        _, _, _, end_mixture = density_precompute_list[-t_idx-2]
        
        curr_mass_ratio = z_start_data.shape[0] / z_end_data.shape[0]
        curr_mass_ratio = torch.tensor(curr_mass_ratio, dtype=torch.float32, device=device)
        
        if support_data:
            manifold_concat = torch.cat((z_start_data, z_end_data, data_support), dim=0)
        else:
            manifold_concat = torch.cat((z_start_data, z_end_data), dim=0)

        current_density_loss = 0
         # Calculate density loss
        if args.backward_density_loss:
            for i in range(len(zt_primal_sample) - 1):
                current_density_loss = current_density_loss + calculate_density_loss(
                    zt_primal_sample[i], 
                    manifold_concat, 
                    k=args.density_k, 
                    hinge_value=args.density_hinge
                )
            # Terminal density loss
            current_density_loss = current_density_loss + calculate_density_loss(
                zt_primal_sample[-1], 
                z_end_data, 
                k=args.exact_density_k, 
                hinge_value=args.exact_density_hinge
            ) * args.terminal_density_weight
            
            density_loss = density_loss + current_density_loss * args.density_coefficient
            
            
        # Calculate MMD loss
        if args.mmd_loss:
            mmd_loss = mmd_loss + mmd_criterion(prediction_data, z_end_data) * args.kl_coefficient
        
        kl_loss = 0
        if args.kl_loss:
            kl_loss = kl_loss + compute_kl_divergence(prediction_data, z_end_data) * args.kl_coefficient
    
        # Calculate PDF loss
        if args.pdf_lossMSE:
            pdf_loss = pdf_loss + compute_pdf_lossMSE(
                start_sample, # initial time query points
                prediction_data, # terminal time query points
                start_logp_diff, # log of Jacobian determinant for initial time
                prediction_logp_diff, # log of Jacobian determinant for terminal time
                start_mixture, # initial time GMM parameters
                end_mixture, # terminal time GMM parameters
                curr_mass_ratio # mass ratio between initial and terminal time
            ) * args.pdf_coefficient
        
        if args.pdf_lossNegLog:
            pdf_loss = pdf_loss + compute_pdf_lossNegLog(
                start_sample, # initial time query points
                prediction_data, # terminal time query points
                start_logp_diff, # log of Jacobian determinant for initial time
                prediction_logp_diff, # log of Jacobian determinant for terminal time
                start_mixture, # initial time GMM parameters
                end_mixture, # terminal time GMM parameters
                curr_mass_ratio # mass ratio between initial and terminal time
            ) * args.pdf_coefficient

        if args.backward_density_loss:
            total_backward_loss += density_loss
        if args.mmd_loss:
            total_backward_loss += mmd_loss
        if args.pdf_lossMSE:
            total_backward_loss += pdf_loss
        if args.pdf_lossMSE:
            total_backward_loss += pdf_loss
        if args.pdf_lossNegLog:
            total_backward_loss += pdf_loss
        if args.kl_loss:
            total_backward_loss += kl_loss
        #if args.manifold_reconstruction_loss:
        #    total_backward_loss += manifold_reconstruction_loss
    
        if args.sync_loss:
            sync_loss = 0 
            consistency_loss = 0
            
            time_length = zt_primal_sample.shape[0]
            mapped_mainfold_list = []
            
            # calculate the consistency loss
            if args.consistency_loss:
                consistency_loss = knn_aux_mean_distance_loss(
                    traj=zt_primal_sample,
                    support_main=total_primal,
                    support_aux=total_secondary,
                    k=30,
                )
                sync_loss = sync_loss + consistency_loss * args.consistency_coefficient
                
            
            mapped_indices, weights, path_indices = map_whole_trajectory2another_manifold(
                traj_points=zt_primal_sample,  # (N, Sample, dim)
                support_points_main=total_primal,
                support_points_aux=total_secondary,
                k=20,
                type=args.map_type
            )
                
            for i in range(time_length):
                if i == 0:
                    mapped_secondary = total_secondary[path_indices[i]]
                    mapped_mainfold_list.append(mapped_secondary)
                else:
                    mapped_secondary = total_secondary[mapped_indices[i]]
                    mapped_mainfold = torch.sum(mapped_secondary * weights[i].unsqueeze(-1), dim=1) 
                    mapped_mainfold_list.append(mapped_mainfold)

            secondary_trajectory = torch.stack(mapped_mainfold_list, dim=0)  # (N, Sample, dim)
            smoothed_secondary_trajectory = smooth_trajectory(secondary_trajectory, window_size=7)

            # secondary energy loss
            secondary_energy_loss = 0
            
        
            
            # generate point cloud pairs
            point_cloud_pairs = [
                (secondary_trajectory[i], secondary_trajectory[i+1])
                for i in range(len(secondary_trajectory) - 1)
            ]

            
            # batch and compute OT_loss
            #batch_losses = [w2_loss((pc1), (pc2)) for pc1, pc2 in point_cloud_pairs]
            batch_losses = [torch.norm(pc1 - pc2, p=2) for pc1, pc2 in point_cloud_pairs]
            secondary_energy_loss = torch.sum(torch.stack(batch_losses))
            sync_loss = sync_loss + secondary_energy_loss * args.sync_energy_coefficient
            
            # smoothness of the secondary trajectory
            
            smoothness_loss = 0
            velocity = secondary_trajectory[1:] - secondary_trajectory[:-1]
            acceleration = velocity[1:] - velocity[:-1]
            smooth_loss = torch.norm(acceleration, p=2)
            smooth_loss = smooth_loss.sum()
            sync_loss = sync_loss + smooth_loss * args.sync_energy_coefficient * 100
            
            
            curr_secondary = secondary_data[t_idx]
            next_secondary = secondary_data[t_idx+1]
            secondary_manifold_concat = torch.cat([curr_secondary, next_secondary], dim=0)
            # secondary manifold loss
            secondary_manifold_loss = 0
            for i in range(len(secondary_trajectory) - 1):
                secondary_manifold_loss = secondary_manifold_loss + calculate_density_loss(
                    smoothed_secondary_trajectory[i], 
                    secondary_manifold_concat, 
                    k=args.density_k, 
                    hinge_value=args.density_hinge
                )
            secondary_manifold_loss = secondary_manifold_loss + calculate_density_loss(
                smoothed_secondary_trajectory[-1], 
                curr_secondary, 
                k=args.exact_density_k, 
                hinge_value=args.exact_density_hinge
            ) * args.terminal_density_weight
                

            
            sync_loss = sync_loss + secondary_manifold_loss * args.density_coefficient * 10000
            
                        # sync pdf loss
            if args.sync_pdf_loss:
                whole_smoothed_points = smoothed_secondary_trajectory.reshape(-1, smoothed_secondary_trajectory.shape[-1])
                sync_pdf_loss = relative_log_pdf(secondary_mixture, whole_smoothed_points, device)
                sync_loss = sync_loss + sync_pdf_loss * args.sync_pdf_coefficient
                
            sync_loss_coefficient = sync_loss * args.sync_coefficient
            

            total_backward_loss += sync_loss_coefficient

            
            
            
            
        
        
        
        # Log losses if writer is provided
    if writer is not None and itr is not None:
        if args.pdf_lossMSE:
            writer.add_scalar('concatenate_long_term_loss/backward_pdf', pdf_loss/args.pdf_coefficient, itr)
        if args.pdf_lossNegLog:
            writer.add_scalar('concatenate_long_term_loss/backward_pdf_neglog', pdf_loss/args.pdf_coefficient, itr)
        if args.backward_density_loss:
            writer.add_scalar('concatenate_long_term_loss/backward_density', density_loss/args.density_coefficient, itr)
        if args.mmd_loss:
            writer.add_scalar('concatenate_long_term_loss/backward_mmd', mmd_loss/args.kl_coefficient, itr)
        if args.kl_loss:
            writer.add_scalar('concatenate_long_term_loss/backward_kl', kl_loss/args.kl_coefficient, itr)
        #if args.manifold_reconstruction_loss:
        #    writer.add_scalar('concatenate_long_term_loss/backward_manifold_reconstruction', manifold_reconstruction_loss/args.manifold_reconstruction_loss_coefficient, itr)
        if args.sync_loss:
            writer.add_scalar('concatenate_long_term_loss/backward_sync', consistency_loss, itr)
            writer.add_scalar('concatenate_long_term_loss/backward_sync_coefficient', sync_loss, itr)
            writer.add_scalar('concatenate_long_term_loss/backward_sync_energy', secondary_energy_loss, itr)
            writer.add_scalar('concatenate_long_term_loss/backward_sync_smoothness', smoothness_loss, itr)
            if args.sync_pdf_loss:
                writer.add_scalar('concatenate_long_term_loss/backward_sync_pdf', sync_pdf_loss, itr)


    return total_backward_loss
                
        
        