

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np




def visualize_evolution(all_samples, terminal_samples=None, save_path='density_evolution.gif', 
                      viz_timesteps=21, fps=10, dpi=100):
    """
    Visualize the evolution of samples over time using animation
    
    Args:
        all_samples (torch.Tensor): Tensor of shape (total_timesteps, num_samples, dim)
        terminal_samples (torch.Tensor, optional): Terminal samples to plot as reference
        save_path (str): Path to save the animation
        viz_timesteps (int): Number of timesteps to visualize
        fps (int): Frames per second for the animation
        dpi (int): Dots per inch for the animation
    """
    # Convert to numpy for visualization
    min_val = all_samples.min()
    max_val = all_samples.max()
    min_val_terminal = terminal_samples.min()
    max_val_terminal = terminal_samples.max()
    xmin = min(min_val, min_val_terminal) - 0.05*abs(min(min_val, min_val_terminal))
    xmax = max(max_val, max_val_terminal) + 0.05*abs(max(max_val, max_val_terminal))
    ymin = min(min_val, min_val_terminal) - 0.05*abs(min(min_val, min_val_terminal))
    ymax = max(max_val, max_val_terminal) + 0.05*abs(max(max_val, max_val_terminal))
    

    if terminal_samples is not None:
        terminal_samples = terminal_samples.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot initial and terminal samples if provided
    if terminal_samples is not None:
        ax.scatter(terminal_samples[:, 0], terminal_samples[:, 1], 
                  color='blue', alpha=0.1, label='Terminal', s=3)
    ax.scatter(all_samples[0, :, 0], all_samples[0, :, 1], 
              color='red', alpha=0.1, label='Initial', s=3)
    
    # Set plot properties
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title('Sample Evolution')
    ax.set_xlabel('X-axis')
    ax.set_aspect('equal')
    ax.grid(False)
    ax.legend()
    
    # Create scatter plots for animation - one for initial points and one for terminal points
    scat_initial = ax.scatter([], [], color='red', alpha=0.7, label='Moving Points')
    if terminal_samples is not None:
        scat_terminal = ax.scatter([], [], color='blue', alpha=0.7, label='Terminal Points')
    
    def init():
        scat_initial.set_offsets(np.empty((0, 2)))
        if terminal_samples is not None:
            scat_terminal.set_offsets(np.empty((0, 2)))
        return (scat_initial,) if terminal_samples is None else (scat_initial, scat_terminal)
    
    def update(frame):
        # Update initial points
        scat_initial.set_offsets(all_samples[frame, :, :])
        scat_initial.set_sizes([0.1] * all_samples.shape[1])
        
        # Update terminal points if provided
        if terminal_samples is not None:
            scat_terminal.set_offsets(terminal_samples)
            scat_terminal.set_sizes([0.1] * terminal_samples.shape[0])
        
        return (scat_initial,) if terminal_samples is None else (scat_initial, scat_terminal)
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=all_samples.shape[0],
                                init_func=init, blit=False, interval=100)
    
    # Save animation with optimized settings
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)
    
    return HTML(ani.to_jshtml())

def plot_trajectory_and_distribution(data_train, all_samples, plot_space=[0,1], n_trajectories=None):
    """
    Plot all trajectories and distributions with smaller point sizes
    
    Args:
        data_train: Training data containing initial and terminal distributions
        all_samples: All trajectory samples
        n_trajectories: Optional parameter to limit number of trajectories (default: plot all)
    """
    plt.figure(figsize=(10, 10))
    dim1 = plot_space[0]
    dim2 = plot_space[1]
    # Plot distributions in blue and red
    n_timepoints = len(data_train)
    colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))
    
    # Plot all time points in data_train
    for i, data in enumerate(data_train):
        plt.scatter(data[:, dim1], data[:, dim2], 
                   alpha=0.3, color=colors[i], 
                   label=f't={i}', s=2)
    
    # Randomly select trajectories
    selected_indices = np.random.choice(all_samples.shape[1], n_trajectories, replace=False)
    
    # Plot trajectories and their points
    for idx in selected_indices:
        trajectory = all_samples[:, idx, :]
        
        # Plot trajectory line in black
        plt.plot(trajectory[:, dim1], trajectory[:, dim2], 
                color='black', alpha=0.5, linewidth=0.5)
        
        # Plot intermediate points with light black dots
        plt.scatter(trajectory[1:-1, dim1], trajectory[1:-1, dim2], 
                   color='black', s=15, alpha=0.2, marker='.')
        
        # Plot initial point with 'x' marker in black
        plt.scatter(trajectory[0, dim1], trajectory[0, dim2], 
                   color='black', s=30, alpha=1, marker='x')
        
        # Plot terminal point with 'o' marker in black
        plt.scatter(trajectory[-1, dim1], trajectory[-1, dim2], 
                   color='black', s=30, alpha=1, marker='^')
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('Evolution Trajectories with Initial and Terminal Distributions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with margin
    all_x = np.concatenate([data_train[0][:, dim1], data_train[1][:, dim1], all_samples[:, :, dim1].flatten()])
    all_y = np.concatenate([data_train[0][:, dim2], data_train[1][:, dim2], all_samples[:, :, dim2].flatten()])
    margin = 0.1
    plt.xlim(all_x.min() - margin, all_x.max() + margin)
    plt.ylim(all_y.min() - margin, all_y.max() + margin)
    
    plt.tight_layout()
    plt.show()


def plot_temporal_embeddings(
    data_list, 
    time_labels=None, 
    colors=None, 
    figsize=(15, 20), 
    alpha=0.5, 
    title="Temporal Embedding Visualization", 
    marker='o', 
    s=30, 
    TwoDplot=True
):
    """
    Plot the temporal embeddings of the data in 2D or 3D.

    Parameters:
        data_list: list of tensors or arrays, each contains the data of a time point
        time_labels: optional, list of time point labels, default is 't0, t1, t2...'
        colors: optional, list of colors, default is the color cycle of matplotlib
        figsize: figure size, default (15, 20)
        alpha: point opacity, default 0.5
        title: figure title, default "Temporal Embedding Visualization"
        marker: scatter plot marker, default 'o'
        s: point size, default 30
        TwoDplot: if True, plot 2D; if False, plot 3D
    Returns:
        fig, ax: matplotlib figure and axis object, can be further customized
    """
    # Check the dimension of the data
    for i, data in enumerate(data_list):
        if TwoDplot:
            if data.shape[1] < 2:
                print(f" Warning: The dataset {i} is not two-dimensional. Shape: {data.shape}")
        else:
            if data.shape[1] < 3:
                print(f" Warning: The dataset {i} is not three-dimensional. Shape: {data.shape}")
    # Select the first 2 or 3 dimensions for plotting
    if TwoDplot:
        data_plot = [data[:, :2] for data in data_list]
    else:
        data_plot = [data[:, :3] for data in data_list]

    # Create default labels if not provided
    if time_labels is None:
        time_labels = [f't{i}' for i in range(len(data_list))]

    # Create default colors list if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        if len(colors) < len(data_list):
            colors = list(mcolors.TABLEAU_COLORS.values()) * ((len(data_list) // len(mcolors.TABLEAU_COLORS)) + 1)
            colors = colors[:len(data_list)]

    # Create the figure and axis
    if TwoDplot:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Plot the data for each time point
    for i, data in enumerate(data_plot):
        color_idx = i % len(colors)  # Cycle through colors if not enough
        data_np = data.detach().cpu().numpy() if hasattr(data, 'detach') else data
        if TwoDplot:
            ax.scatter(data_np[:, 0], data_np[:, 1], 
                       color=colors[color_idx], 
                       alpha=alpha, 
                       label=time_labels[i],
                       marker=marker,
                       s=s)
        else:
            ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2],
                       color=colors[color_idx], 
                       alpha=alpha, 
                       label=time_labels[i],
                       marker=marker,
                       s=s)

    # Add legend and labels
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    if not TwoDplot:
        ax.set_zlabel("Dimension 3", fontsize=12)
        ax.set_box_aspect([1,1,1])  # Keep xyz aspect ratio equal
        # Optionally, you can set the view angle:
        # ax.view_init(elev=20, azim=120)
    else:
        ax.set_aspect('equal')

    return fig, ax