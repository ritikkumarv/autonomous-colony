"""
Advanced Visualization and Rendering for The Autonomous Colony

Provides rich visualizations:
- Interactive grid rendering with agents and resources
- Communication flow visualization
- Training metrics and dashboards
- Heatmaps for exploration and activity
- Trajectory tracking and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque


class GridRenderer:
    """
    Renders the colony grid with agents and resources.
    
    Features:
    - Color-coded resources
    - Agent markers with IDs and health
    - Grid overlays
    - Real-time updates
    """
    
    def __init__(self, grid_size: int = 20, figsize: Tuple[int, int] = (12, 10)):
        self.grid_size = grid_size
        self.figsize = figsize
        
        # Color scheme
        self.colors = {
            'empty': np.array([0.1, 0.1, 0.1]),
            'food': np.array([0.2, 0.8, 0.2]),
            'water': np.array([0.2, 0.5, 1.0]),
            'material': np.array([0.6, 0.4, 0.2]),
            'obstacle': np.array([0.3, 0.3, 0.3])
        }
        
        # Agent colors (colorblind-friendly palette)
        self.agent_colors = plt.cm.Set2(np.linspace(0, 1, 10))
        
        self.fig = None
        self.ax = None
    
    def render(
        self,
        grid: np.ndarray,
        agents: List,
        step: int = 0,
        show_stats: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Render the current state of the environment.
        
        Args:
            grid: Grid array with resource types
            agents: List of agent objects
            step: Current step number
            show_stats: Whether to show agent statistics
            save_path: Optional path to save figure
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
            gs = GridSpec(1, 2, width_ratios=[2, 1], figure=self.fig)
            self.ax = self.fig.add_subplot(gs[0])
            self.stats_ax = self.fig.add_subplot(gs[1])
        else:
            self.ax.clear()
            self.stats_ax.clear()
        
        # Render grid
        grid_vis = self._create_grid_visualization(grid)
        self.ax.imshow(grid_vis, interpolation='nearest')
        
        # Render agents
        self._render_agents(agents)
        
        # Configure axes
        self.ax.set_title(f'Colony World - Step {step}', fontsize=14, fontweight='bold')
        self.ax.set_xticks(np.arange(0, self.grid_size, 5))
        self.ax.set_yticks(np.arange(0, self.grid_size, 5))
        self.ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Render stats
        if show_stats:
            self._render_stats(agents, step)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.pause(0.01)
    
    def _create_grid_visualization(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to RGB visualization."""
        h, w = grid.shape
        grid_vis = np.zeros((h, w, 3))
        
        # Map resource types to colors
        resource_types = {
            0: 'empty',
            1: 'food',
            2: 'water',
            3: 'material',
            4: 'obstacle'
        }
        
        for resource_type, color_name in resource_types.items():
            mask = grid == resource_type
            grid_vis[mask] = self.colors[color_name]
        
        return grid_vis
    
    def _render_agents(self, agents: List):
        """Render agents on the grid."""
        for i, agent in enumerate(agents):
            if not agent.alive:
                continue
            
            x, y = agent.position.x, agent.position.y
            
            # Agent circle
            color = self.agent_colors[i % len(self.agent_colors)]
            circle = plt.Circle(
                (x, y),
                radius=0.4,
                color=color,
                ec='white',
                linewidth=2,
                zorder=10
            )
            self.ax.add_patch(circle)
            
            # Agent ID
            self.ax.text(
                x, y,
                str(i),
                ha='center',
                va='center',
                color='white',
                fontsize=10,
                fontweight='bold',
                zorder=11
            )
            
            # Health bar
            health_ratio = agent.stats.health / 100.0
            if health_ratio < 1.0:
                bar_width = 0.8
                bar_height = 0.1
                bar_x = x - bar_width / 2
                bar_y = y + 0.6
                
                # Background (red)
                bg_rect = patches.Rectangle(
                    (bar_x, bar_y),
                    bar_width,
                    bar_height,
                    linewidth=0,
                    facecolor='red',
                    zorder=9
                )
                self.ax.add_patch(bg_rect)
                
                # Health (green)
                health_rect = patches.Rectangle(
                    (bar_x, bar_y),
                    bar_width * health_ratio,
                    bar_height,
                    linewidth=0,
                    facecolor='green',
                    zorder=10
                )
                self.ax.add_patch(health_rect)
    
    def _render_stats(self, agents: List, step: int):
        """Render agent statistics panel."""
        self.stats_ax.axis('off')
        
        stats_text = f"COLONY STATUS\nStep {step}\n" + "="*30 + "\n\n"
        
        alive_count = sum(1 for a in agents if a.alive)
        stats_text += f"Active Agents: {alive_count}/{len(agents)}\n\n"
        
        for i, agent in enumerate(agents):
            status = "🟢" if agent.alive else "🔴"
            stats_text += f"{status} Agent {i}\n"
            
            if agent.alive:
                stats_text += f"  Pos: ({agent.position.x}, {agent.position.y})\n"
                stats_text += f"  Energy: {agent.stats.energy:.1f}/100\n"
                stats_text += f"  Health: {agent.stats.health:.1f}/100\n"
                stats_text += f"  Food: {agent.stats.food_count}\n"
                stats_text += f"  Water: {agent.stats.water_count}\n"
                stats_text += f"  Materials: {agent.stats.material_count}\n"
            else:
                stats_text += f"  DEAD\n"
            
            stats_text += "\n"
        
        self.stats_ax.text(
            0.05, 0.95,
            stats_text,
            transform=self.stats_ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
    
    def close(self):
        """Close the figure."""
        if self.fig:
            plt.close(self.fig)


class CommunicationVisualizer:
    """
    Visualizes communication between agents.
    
    Shows message passing as arrows/lines between agents.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def render(
        self,
        agent_positions: List[Tuple[int, int]],
        messages: np.ndarray,
        grid_size: int = 20
    ):
        """
        Render communication flow between agents.
        
        Args:
            agent_positions: List of (x, y) positions
            messages: Communication matrix (n_agents, n_agents, message_dim)
            grid_size: Size of the grid
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        else:
            self.ax.clear()
        
        # Draw agents
        for i, (x, y) in enumerate(agent_positions):
            circle = plt.Circle(
                (x, y),
                radius=0.5,
                color=plt.cm.Set2(i / len(agent_positions)),
                ec='black',
                linewidth=2
            )
            self.ax.add_patch(circle)
            self.ax.text(x, y, str(i), ha='center', va='center', fontweight='bold')
        
        # Draw communication arrows
        if messages is not None:
            # Compute message strengths (norm of message vectors)
            message_strengths = np.linalg.norm(messages, axis=2)
            max_strength = message_strengths.max()
            
            for i in range(len(agent_positions)):
                for j in range(len(agent_positions)):
                    if i == j:
                        continue
                    
                    strength = message_strengths[i, j]
                    if strength > 0.1 * max_strength:  # Only show significant messages
                        x1, y1 = agent_positions[i]
                        x2, y2 = agent_positions[j]
                        
                        # Arrow properties
                        alpha = min(strength / max_strength, 1.0)
                        width = 0.5 + 2.0 * (strength / max_strength)
                        
                        self.ax.arrow(
                            x1, y1,
                            x2 - x1, y2 - y1,
                            head_width=0.5,
                            head_length=0.3,
                            fc='blue',
                            ec='blue',
                            alpha=alpha,
                            linewidth=width,
                            length_includes_head=True
                        )
        
        self.ax.set_xlim(-1, grid_size)
        self.ax.set_ylim(-1, grid_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('Agent Communication Flow', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)


class HeatmapVisualizer:
    """
    Creates heatmaps for various metrics:
    - Exploration (where agents have been)
    - Resource collection hotspots
    - Agent activity
    """
    
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.exploration_map = np.zeros((grid_size, grid_size))
        self.collection_map = np.zeros((grid_size, grid_size))
        self.activity_map = np.zeros((grid_size, grid_size))
    
    def update(self, agent_positions: List[Tuple[int, int]], collected_at: Optional[List[Tuple[int, int]]] = None):
        """Update heatmaps with new data."""
        # Update exploration
        for x, y in agent_positions:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.exploration_map[y, x] += 1
                self.activity_map[y, x] += 1
        
        # Update collection hotspots
        if collected_at:
            for x, y in collected_at:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.collection_map[y, x] += 1
    
    def render(self, save_path: Optional[str] = None):
        """Render all heatmaps."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Exploration heatmap
        sns.heatmap(
            self.exploration_map,
            ax=axes[0],
            cmap='YlOrRd',
            cbar_kws={'label': 'Visits'},
            square=True
        )
        axes[0].set_title('Exploration Heatmap', fontweight='bold')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        
        # Collection heatmap
        sns.heatmap(
            self.collection_map,
            ax=axes[1],
            cmap='Blues',
            cbar_kws={'label': 'Collections'},
            square=True
        )
        axes[1].set_title('Resource Collection Heatmap', fontweight='bold')
        axes[1].set_xlabel('X Position')
        axes[1].set_ylabel('Y Position')
        
        # Activity heatmap
        sns.heatmap(
            self.activity_map,
            ax=axes[2],
            cmap='Greens',
            cbar_kws={'label': 'Activity'},
            square=True
        )
        axes[2].set_title('Agent Activity Heatmap', fontweight='bold')
        axes[2].set_xlabel('X Position')
        axes[2].set_ylabel('Y Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
    
    def reset(self):
        """Reset all heatmaps."""
        self.exploration_map.fill(0)
        self.collection_map.fill(0)
        self.activity_map.fill(0)


class TrajectoryVisualizer:
    """
    Tracks and visualizes agent trajectories over time.
    """
    
    def __init__(self, n_agents: int, grid_size: int = 20, max_history: int = 100):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.max_history = max_history
        
        # Store trajectories
        self.trajectories = [deque(maxlen=max_history) for _ in range(n_agents)]
        self.colors = plt.cm.Set2(np.linspace(0, 1, n_agents))
    
    def add_positions(self, positions: List[Tuple[int, int]]):
        """Add new positions to trajectories."""
        for i, pos in enumerate(positions):
            if i < self.n_agents:
                self.trajectories[i].append(pos)
    
    def render(self, current_grid: Optional[np.ndarray] = None, save_path: Optional[str] = None):
        """Render trajectories on grid."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Background grid (if provided)
        if current_grid is not None:
            ax.imshow(
                np.ones_like(current_grid) * 0.9,
                cmap='gray',
                alpha=0.3,
                extent=[0, self.grid_size, 0, self.grid_size]
            )
        
        # Draw trajectories
        for i, trajectory in enumerate(self.trajectories):
            if len(trajectory) < 2:
                continue
            
            traj_array = np.array(trajectory)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]
            
            # Plot trajectory line
            ax.plot(
                x_coords, y_coords,
                color=self.colors[i],
                linewidth=2,
                alpha=0.7,
                label=f'Agent {i}'
            )
            
            # Plot trajectory points (fading older ones)
            for j, (x, y) in enumerate(trajectory):
                alpha = 0.3 + 0.7 * (j / len(trajectory))
                ax.scatter(x, y, color=self.colors[i], s=20, alpha=alpha, zorder=5)
            
            # Mark current position
            if len(trajectory) > 0:
                x, y = trajectory[-1]
                ax.scatter(x, y, color=self.colors[i], s=100, marker='o', 
                          edgecolor='white', linewidth=2, zorder=10)
                ax.text(x, y, str(i), ha='center', va='center',
                       color='white', fontweight='bold', fontsize=9)
        
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.set_title('Agent Trajectories', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
    
    def reset(self):
        """Clear all trajectories."""
        for traj in self.trajectories:
            traj.clear()


class TrainingDashboard:
    """
    Real-time training metrics dashboard.
    
    Shows:
    - Episode rewards over time
    - Success rate
    - Agent statistics
    - Loss curves
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Metrics storage
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.success_rates = deque(maxlen=window_size)
        self.losses = defaultdict(lambda: deque(maxlen=window_size))
        
        self.fig = None
        self.axes = None
    
    def update(self, metrics: Dict):
        """Update dashboard with new metrics."""
        if 'episode_reward' in metrics:
            self.episode_rewards.append(metrics['episode_reward'])
        if 'episode_length' in metrics:
            self.episode_lengths.append(metrics['episode_length'])
        if 'success' in metrics:
            self.success_rates.append(1.0 if metrics['success'] else 0.0)
        
        # Update losses
        for key, value in metrics.items():
            if 'loss' in key.lower():
                self.losses[key].append(value)
    
    def render(self, save_path: Optional[str] = None):
        """Render the dashboard."""
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
            self.fig.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
        
        for ax in self.axes.flat:
            ax.clear()
        
        # Episode rewards
        if len(self.episode_rewards) > 0:
            self.axes[0, 0].plot(self.episode_rewards, linewidth=2, color='blue', alpha=0.6)
            self.axes[0, 0].set_title('Episode Rewards')
            self.axes[0, 0].set_xlabel('Episode')
            self.axes[0, 0].set_ylabel('Total Reward')
            self.axes[0, 0].grid(True, alpha=0.3)
            
            # Moving average
            if len(self.episode_rewards) > 10:
                ma = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                self.axes[0, 0].plot(range(9, len(ma)+9), ma, 
                                    linewidth=2, color='red', label='MA(10)')
                self.axes[0, 0].legend()
        
        # Success rate
        if len(self.success_rates) > 0:
            # Moving average of success rate
            window = min(20, len(self.success_rates))
            if len(self.success_rates) >= window:
                ma_success = np.convolve(self.success_rates, np.ones(window)/window, mode='valid')
                self.axes[0, 1].plot(range(window-1, len(ma_success)+window-1), ma_success,
                                    linewidth=2, color='green')
            self.axes[0, 1].set_title(f'Success Rate (MA {window})')
            self.axes[0, 1].set_xlabel('Episode')
            self.axes[0, 1].set_ylabel('Success Rate')
            self.axes[0, 1].set_ylim([0, 1])
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Episode lengths
        if len(self.episode_lengths) > 0:
            self.axes[1, 0].plot(self.episode_lengths, linewidth=2, color='purple', alpha=0.6)
            self.axes[1, 0].set_title('Episode Lengths')
            self.axes[1, 0].set_xlabel('Episode')
            self.axes[1, 0].set_ylabel('Steps')
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Losses
        if len(self.losses) > 0:
            for loss_name, loss_values in self.losses.items():
                if len(loss_values) > 0:
                    self.axes[1, 1].plot(loss_values, label=loss_name, linewidth=2, alpha=0.7)
            self.axes[1, 1].set_title('Training Losses')
            self.axes[1, 1].set_xlabel('Update Step')
            self.axes[1, 1].set_ylabel('Loss')
            self.axes[1, 1].legend(fontsize=8)
            self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.pause(0.01)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Visualization Components...\n")
    
    # Test GridRenderer
    print("1. Testing GridRenderer:")
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.environment import ColonyEnvironment
    
    env = ColonyEnvironment(n_agents=3, grid_size=20)
    env.reset()
    
    renderer = GridRenderer(grid_size=20)
    renderer.render(env.world.grid, env.agents, step=0, save_path='test_grid.png')
    print(f"   ✓ Grid rendered and saved to test_grid.png\n")
    renderer.close()
    
    # Test HeatmapVisualizer
    print("2. Testing HeatmapVisualizer:")
    heatmap_viz = HeatmapVisualizer(grid_size=20)
    
    # Simulate some activity
    for _ in range(100):
        positions = [(np.random.randint(0, 20), np.random.randint(0, 20)) for _ in range(3)]
        heatmap_viz.update(positions)
    
    heatmap_viz.render(save_path='test_heatmaps.png')
    print(f"   ✓ Heatmaps rendered and saved to test_heatmaps.png\n")
    
    # Test TrajectoryVisualizer
    print("3. Testing TrajectoryVisualizer:")
    traj_viz = TrajectoryVisualizer(n_agents=3, grid_size=20)
    
    # Simulate random walk
    for step in range(50):
        positions = []
        for agent_id in range(3):
            x = int(10 + 5 * np.sin(step * 0.1 + agent_id))
            y = int(10 + 5 * np.cos(step * 0.1 + agent_id))
            positions.append((x, y))
        traj_viz.add_positions(positions)
    
    traj_viz.render(save_path='test_trajectories.png')
    print(f"   ✓ Trajectories rendered and saved to test_trajectories.png\n")
    
    # Test TrainingDashboard
    print("4. Testing TrainingDashboard:")
    dashboard = TrainingDashboard(window_size=100)
    
    # Simulate training
    for episode in range(100):
        metrics = {
            'episode_reward': 50 + 30 * np.sin(episode * 0.1) + np.random.randn() * 10,
            'episode_length': 200 + np.random.randint(-50, 50),
            'success': np.random.rand() < (0.3 + episode * 0.005),
            'policy_loss': 0.5 * np.exp(-episode * 0.01) + np.random.rand() * 0.1,
            'value_loss': 1.0 * np.exp(-episode * 0.01) + np.random.rand() * 0.2
        }
        dashboard.update(metrics)
    
    dashboard.render(save_path='test_dashboard.png')
    print(f"   ✓ Dashboard rendered and saved to test_dashboard.png\n")
    
    print("✅ All visualization tests passed!")
    print("\nGenerated files:")
    print("  - test_grid.png")
    print("  - test_heatmaps.png")
    print("  - test_trajectories.png")
    print("  - test_dashboard.png")
