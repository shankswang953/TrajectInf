import argparse

def get_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    parser = argparse.ArgumentParser(description='Configuration for trajectory inference model')
    parser.add_argument('--UnbalancedOT', type=bool, default=False)
    
    # General settings
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    # Model architecture
    parser.add_argument('--width', type=int, default=64, help='Width of the network')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimensionality of hidden layers')
    parser.add_argument('--n_hiddens', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('--otdim', type=int, default=5, help='Output dimension')
    
    # Training parameters
    parser.add_argument('--niters', type=int, default=3000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--noise_initial_sample', type=float, default=0.35, help='Initial noise level for sampling')
    
    # Method options
    parser.add_argument('--map_type', type=str, default='Simple', help='Type of map')
    
    # Directories
    parser.add_argument('--train_dir', type=str, default='./checkpoint', help='Training directory for checkpoints')
    parser.add_argument('--results_dir', type=str, default='./MultiState', help='Directory to store results')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='../../data/scMultiSim/all_time_scRNA_pca.npz', help='Path to data file')
    parser.add_argument('--time_labels', type=str, nargs='+', default=['time1', 'time2', 'time3', 'time4'], # the last one is the support of the bump
                        help='Time point labels')
    parser.add_argument('--support_points', type=bool, default=False, help='Use support points')
    parser.add_argument('--time_points', type=int, nargs='+', default=[1, 2, 3, 4], 
                        help='Time points for integration')
    parser.add_argument('--time_steps', type=int, nargs='+', default=[1, 1, 1], 
                        help='Time steps for integration')
    parser.add_argument('--long_time_steps', type=int, nargs='+', default=[1, 2, 3], 
                        help='Long time steps for integration')
    parser.add_argument('--activation', type=str, default='leakyrelu')
    # Hyperparameters for density loss
    parser.add_argument('--dt', type=float, default=0.2, help='Time step for density loss')
    parser.add_argument('--density_k', type=int, default=20, help='Number of backward points')
    parser.add_argument('--exact_density_k', type=int, default=20, help='Number of terminal backward points')
    parser.add_argument('--density_hinge', type=float, default=0.005, help='Hinge value for backward points')
    parser.add_argument('--exact_density_hinge', type=float, default=0.005, help='Hinge value for terminal backward points')
    parser.add_argument('--terminal_density_weight', type=float, default=5.0, help='Weight for terminal density loss')
    # Loss parameters dictionary
    loss_group = parser.add_argument_group('Loss Parameters')
    loss_group.add_argument('--manifold_reconstruction_loss', type=bool, default=False, help='Use manifold reconstruction loss')
    loss_group.add_argument('--manifold_reconstruction_loss_coefficient', type=float, default=5, help='Coefficient for manifold reconstruction loss')
    loss_group.add_argument('--mmd_loss', type=bool, default=False, help='Use MMD loss')
    loss_group.add_argument('--manifold_logdensity_loss', type=bool, default=True, help='Use manifold logdensity loss')
    loss_group.add_argument('--adaptive_sigma', type=bool, default=False, help='Use adaptive sigma')
    loss_group.add_argument('--long_term_loss', type=bool, default=False, help='Use long term loss')
    loss_group.add_argument('--energy_g_weight', type=float, default=0.0, help='Weight for energy loss')
    loss_group.add_argument('--forward_density_loss', type=bool, default=False, help='Use forward density loss')
    loss_group.add_argument('--backward_density_loss', type=bool, default=True, help='Use backward density loss')
    loss_group.add_argument('--density_coefficient', type=float, default=2000.0, help='Coefficient for density loss')
    loss_group.add_argument('--pdf_lossMSE', type=bool, default=False, help='Use probability loss')
    loss_group.add_argument('--pdf_lossNegLog', type=bool, default=True, help='Use probability loss')
    loss_group.add_argument('--pdf_coefficient', type=float, default=2000.0, help='Coefficient for probability loss')
    loss_group.add_argument('--pdf_bandwidth', type=float, default=0.01, help='Bandwidth for probability loss')
    loss_group.add_argument('--sigma', type=float, default=0.0005, help='Sigma for probability loss')
    loss_group.add_argument('--kl_loss', type=bool, default=True, help='Use KL divergence loss')
    loss_group.add_argument('--kl_coefficient', type=float, default=60000.0, help='Coefficient for KL divergence loss')
    loss_group.add_argument('--forward_energy_coefficient', type=float, default=300.0, help='Coefficient for forward energy loss')
    loss_group.add_argument('--geo_loss', type=bool, default=False, help='Use backward geodesic loss')
    loss_group.add_argument('--forward_geo_loss', type=bool, default=False, help='Use forward geodesic loss')
    loss_group.add_argument('--geo_coefficient', type=float, default=200.0, help='Coefficient for geodesic loss')
    
    loss_group.add_argument('--concatenate_loss', type=bool, default=False, help='Use concatenate loss')
    loss_group.add_argument('--concatenate_coefficient', type=float, default=1.0, help='Coefficient for concatenate loss')
    
    
    # synchronization parameters
    loss_group.add_argument('--sync_loss', type=bool, default=True, help='Use synchronization ot loss')
    loss_group.add_argument('--sync_coefficient', type=float, default=10.0, help='Coefficient for synchronization ot loss')
    parser.add_argument('--sync_data_path', type=str, default='../../data/scMultiSim/all_time_scATAC_pca.npz', help='Path to synchronization data file')
    
    loss_group.add_argument('--sync_mmd_loss', type=bool, default=True, help='Use synchronization mmd loss')
    loss_group.add_argument('--sync_mmd_coefficient', type=float, default=5000.0, help='Coefficient for synchronization mmd loss')
    loss_group.add_argument('--sync_energy_coefficient', type=float, default=10.0, help='Coefficient for synchronization energy loss')

    loss_group.add_argument('--consistency_loss', type=bool, default=False, help='Use consistency loss')
    parser.add_argument('--consistency_coefficient', type=float, default=10.0, help='Coefficient for consistency loss')
    
    
    loss_group.add_argument('--sync_manifold_loss', type=bool, default=False, help='Use synchronization manifold loss')
    
    loss_group.add_argument('--sync_pdf_loss', type=bool, default=True, help='Use synchronization pdf loss')
    loss_group.add_argument('--sync_pdf_coefficient', type=float, default=1000.0, help='Coefficient for synchronization pdf loss')
    loss_group.add_argument('--sync_kl_divergence_loss', type=bool, default=True, help='Use synchronization kl divergence loss')
    
    
    return parser.parse_args()

def get_notebook_args():
    """
    Get arguments for Jupyter notebook environment (no command line parsing)
    
    Returns:
        argparse.Namespace: Default arguments for notebook use
    """
    parser = argparse.ArgumentParser(description='Configuration for trajectory inference model')
    
    # General settings
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--viz', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--UnbalancedOT', type=bool, default=False)
    
    # Model architecture
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_hiddens', type=int, default=5)
    parser.add_argument('--otdim', type=int, default=5)
    
    # Training parameters
    parser.add_argument('--niters', type=int, default=800)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--noise_initial_sample', type=float, default=0.35)
    
    
    parser.add_argument('--sigma', type=float, default=0.0005)
    parser.add_argument('--dt', type=float, default=0.05)
    
    # Method options
    
    # Directories
    parser.add_argument('--train_dir', type=str, default='./checkpoint')
    parser.add_argument('--results_dir', type=str, default='./MultiState')
    
    # Method options
    parser.add_argument('--map_type', type=str, default='Simple')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='../../data/scMultiSim/all_time_scRNA_pca.npz')
    parser.add_argument('--time_labels', type=str, nargs='+', default=['time1', 'time2', 'time3', 'time4'])# the last one is the support of the bump
    parser.add_argument('--support_points', type=bool, default=False)
    parser.add_argument('--time_points', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--time_steps', type=int, nargs='+', default=[1, 1, 1])
    parser.add_argument('--long_time_steps', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--activation', type=str, default='leakyrelu')
    
    parser.add_argument('--sync_data_path', type=str, default='../../data/scMultiSim/all_time_scATAC_pca.npz')
    
    return parser.parse_args([])

# For backward compatibility
class Args:
    """Legacy Args class for backward compatibility"""
    def __init__(self):
        args = get_args()
        for key, value in vars(args).items():
            setattr(self, key, value)
            
if __name__ == "__main__":
    args = get_args()
    print(args)