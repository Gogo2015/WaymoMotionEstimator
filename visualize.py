import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_trajectory(past_xy,true_future_xy,pred_future_xy,save_path):
    """
    Create a GIF comparing predicted vs true path.

    past_xy : (T_past, 2)
    true_future_xy : (T_future, 2)
    pred_future_xy : (T_future, 2)
    save_path : str
    """

    title="Agent trajectory"
    past_label="past"
    true_label="true future"
    pred_label="pred future"
    fps=8
    pad=5.0
    figsize=(4, 4)

    past_xy = np.asarray(past_xy)
    true_future_xy = np.asarray(true_future_xy)
    pred_future_xy = np.asarray(pred_future_xy)

    assert past_xy.ndim == 2 and past_xy.shape[1] == 2
    assert true_future_xy.shape == pred_future_xy.shape
    T_future = true_future_xy.shape[0]

    # global bounds so every frame has same window
    all_pts = np.concatenate([past_xy, true_future_xy, pred_future_xy], axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)

    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)

    # show past
    past_line, = ax.plot(
        past_xy[:, 0],
        past_xy[:, 1],
        "o-",
        color="blue",
        linewidth=1.5,
        markersize=3,
        label=past_label,
    )

    # animated true/pred
    true_line, = ax.plot([], [], "o-", color="green", linewidth=1.5, markersize=3, label=true_label)
    pred_line, = ax.plot([], [], "o--", color="red", linewidth=1.5, markersize=3, label=pred_label)

    # legend
    ax.legend(loc="upper right", fontsize=6, frameon=True)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    def init():
        true_line.set_data([], [])
        pred_line.set_data([], [])
        return true_line, pred_line

    def update(frame):
        # frame 0..T_future-1
        true_line.set_data(true_future_xy[:frame+1, 0], true_future_xy[:frame+1, 1])
        pred_line.set_data(pred_future_xy[:frame+1, 0], pred_future_xy[:frame+1, 1])
        return true_line, pred_line

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=T_future,
        interval=1000 / fps,
        blit=True,
    )

    anim.save(save_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

def visualize_multimodal_trajectory(past_xy, true_future_xy, pred_futures_xy, confidences, save_path):
    """
    Create a GIF showing multiple predicted trajectories with confidences.
    
    Args:
        past_xy: (T_past, 2)
        true_future_xy: (T_future, 2)
        pred_futures_xy: (K, T_future, 2) - K predicted trajectories
        confidences: (K,) - confidence for each mode
        save_path: str
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    
    past_xy = np.asarray(past_xy)
    true_future_xy = np.asarray(true_future_xy)
    pred_futures_xy = np.asarray(pred_futures_xy)
    confidences = np.asarray(confidences)
    
    K = pred_futures_xy.shape[0]  # number of modes
    T_future = true_future_xy.shape[0]
    
    # Global bounds
    all_pts = [past_xy, true_future_xy] + [pred_futures_xy[k] for k in range(K)]
    all_pts = np.concatenate(all_pts, axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    
    pad = 5.0
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title("Multi-Modal Trajectory Prediction", fontsize=12)
    
    # Show past
    ax.plot(past_xy[:, 0], past_xy[:, 1], "o-", color="blue", 
            linewidth=2, markersize=4, label="past", zorder=10)
    
    # Colors for different modes
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Static: show all predicted trajectories as faint lines
    for k in range(K):
        ax.plot(pred_futures_xy[k, :, 0], pred_futures_xy[k, :, 1], 
                '--', color=colors[k], linewidth=1, alpha=0.3)
    
    # Animated lines for each mode
    pred_lines = []
    for k in range(K):
        line, = ax.plot([], [], "o--", color=colors[k], linewidth=2, 
                       markersize=3, alpha=0.7,
                       label=f"mode {k+1} ({confidences[k]:.2f})")
        pred_lines.append(line)
    
    # Animated true trajectory
    true_line, = ax.plot([], [], "o-", color="green", linewidth=2.5, 
                        markersize=4, label="ground truth", zorder=5)
    
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    
    def init():
        true_line.set_data([], [])
        for line in pred_lines:
            line.set_data([], [])
        return [true_line] + pred_lines
    
    def update(frame):
        # Update true trajectory
        true_line.set_data(true_future_xy[:frame+1, 0], true_future_xy[:frame+1, 1])
        
        # Update all predicted modes
        for k, line in enumerate(pred_lines):
            line.set_data(pred_futures_xy[k, :frame+1, 0], 
                         pred_futures_xy[k, :frame+1, 1])
        
        return [true_line] + pred_lines
    
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=T_future,
        interval=125, blit=True  # 8 fps
    )
    
    anim.save(save_path, writer=animation.PillowWriter(fps=8))
    plt.close(fig)
    print(f"Saved multi-modal visualization to {save_path}")
