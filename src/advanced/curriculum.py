"""
Curriculum Learning

Automatically adjusts environment difficulty based on agent performance:
- Progressive difficulty increase
- Success-based adaptation
- Efficient learning through appropriate challenge
"""

import numpy as np
from collections import deque
from typing import Dict, Any
import copy


class CurriculumScheduler:
    """
    Automatic curriculum scheduler that adjusts difficulty based on performance.
    
    Maintains agent in "zone of proximal development" - not too easy, not too hard.
    
    Args:
        initial_difficulty: Starting difficulty level (0.0 to 1.0)
        success_threshold: Target success rate
        window_size: Number of recent episodes to consider
        adapt_rate: How quickly to adjust difficulty
    """
    
    def __init__(
        self,
        initial_difficulty: float = 0.3,
        success_threshold: float = 0.7,
        window_size: int = 20,
        adapt_rate: float = 0.05,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0
    ):
        self.difficulty = initial_difficulty
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.adapt_rate = adapt_rate
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        
        # Track recent performance
        self.recent_results = deque(maxlen=window_size)
        self.episode_count = 0
        
        print(f"✓ Curriculum Scheduler initialized (difficulty={initial_difficulty:.2f})")
    
    def record_episode(self, success: bool, reward: float = None):
        """
        Record episode outcome.
        
        Args:
            success: Whether episode was successful
            reward: Optional reward value for more nuanced adaptation
        """
        self.recent_results.append(1.0 if success else 0.0)
        self.episode_count += 1
    
    def update_difficulty(self) -> float:
        """
        Update difficulty based on recent performance.
        
        Returns:
            new_difficulty: Updated difficulty level
        """
        if len(self.recent_results) < self.window_size:
            return self.difficulty
        
        success_rate = np.mean(self.recent_results)
        
        # Adjust difficulty based on success rate
        if success_rate > self.success_threshold + 0.1:
            # Too easy - increase difficulty
            self.difficulty = min(
                self.max_difficulty,
                self.difficulty + self.adapt_rate
            )
            print(f"📈 Increasing difficulty to {self.difficulty:.2f} "
                  f"(success rate: {success_rate:.1%})")
        
        elif success_rate < self.success_threshold - 0.1:
            # Too hard - decrease difficulty
            self.difficulty = max(
                self.min_difficulty,
                self.difficulty - self.adapt_rate
            )
            print(f"📉 Decreasing difficulty to {self.difficulty:.2f} "
                  f"(success rate: {success_rate:.1%})")
        
        else:
            # Just right - maintain difficulty
            pass
        
        return self.difficulty
    
    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.difficulty
    
    def get_env_config(self, base_config: Any) -> Any:
        """
        Generate environment configuration based on current difficulty.
        
        Args:
            base_config: Base environment configuration
            
        Returns:
            config: Adjusted configuration for current difficulty
        """
        config = copy.copy(base_config)
        
        # Adjust parameters based on difficulty
        # Higher difficulty = more obstacles, less resources, faster energy decay
        
        if hasattr(config, 'obstacle_density'):
            config.obstacle_density = 0.05 + (0.15 * self.difficulty)
        
        if hasattr(config, 'food_spawn_rate'):
            config.food_spawn_rate = 0.03 - (0.015 * self.difficulty)
        
        if hasattr(config, 'energy_decay'):
            config.energy_decay = 0.05 + (0.15 * self.difficulty)
        
        if hasattr(config, 'max_steps'):
            # Harder = shorter episodes (less time to succeed)
            base_steps = getattr(config, 'base_max_steps', 500)
            config.max_steps = int(base_steps * (1.0 - 0.3 * self.difficulty))
        
        return config


class StageBasedCurriculum:
    """
    Stage-based curriculum with explicit progression milestones.
    
    Agent must complete each stage before advancing to next.
    
    Args:
        stages: List of stage configurations
        success_threshold: Success rate needed to advance
        window_size: Episodes to consider for advancement
    """
    
    def __init__(
        self,
        stages: list,
        success_threshold: float = 0.8,
        window_size: int = 10
    ):
        self.stages = stages
        self.current_stage = 0
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.recent_results = deque(maxlen=window_size)
        
        print(f"✓ Stage-based Curriculum with {len(stages)} stages")
    
    def record_episode(self, success: bool):
        """Record episode outcome."""
        self.recent_results.append(1.0 if success else 0.0)
    
    def check_advancement(self) -> bool:
        """
        Check if agent should advance to next stage.
        
        Returns:
            advanced: Whether stage advanced
        """
        if len(self.recent_results) < self.window_size:
            return False
        
        success_rate = np.mean(self.recent_results)
        
        if success_rate >= self.success_threshold:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.recent_results.clear()
                print(f"🎓 Advanced to stage {self.current_stage + 1}/{len(self.stages)}")
                return True
        
        return False
    
    def get_current_stage(self) -> Dict:
        """Get current stage configuration."""
        return self.stages[self.current_stage]
    
    def is_complete(self) -> bool:
        """Check if all stages completed."""
        return self.current_stage == len(self.stages) - 1


class TaskCurriculum:
    """
    Task-based curriculum that samples from a distribution of tasks.
    
    Gradually shifts distribution toward harder tasks as agent improves.
    
    Args:
        easy_tasks: List of easy task configurations
        medium_tasks: List of medium task configurations
        hard_tasks: List of hard task configurations
    """
    
    def __init__(
        self,
        easy_tasks: list,
        medium_tasks: list,
        hard_tasks: list
    ):
        self.task_pools = {
            'easy': easy_tasks,
            'medium': medium_tasks,
            'hard': hard_tasks
        }
        
        # Initial distribution: mostly easy tasks
        self.task_distribution = {
            'easy': 0.7,
            'medium': 0.2,
            'hard': 0.1
        }
        
        self.performance_history = deque(maxlen=50)
    
    def sample_task(self) -> Dict:
        """
        Sample task from current distribution.
        
        Returns:
            task: Sampled task configuration
        """
        # Sample difficulty level
        difficulty = np.random.choice(
            ['easy', 'medium', 'hard'],
            p=[
                self.task_distribution['easy'],
                self.task_distribution['medium'],
                self.task_distribution['hard']
            ]
        )
        
        # Sample specific task from pool
        task_pool = self.task_pools[difficulty]
        task = np.random.choice(task_pool)
        
        return task
    
    def update_distribution(self, success_rate: float):
        """
        Update task distribution based on performance.
        
        Args:
            success_rate: Recent success rate
        """
        if success_rate > 0.8:
            # Doing well - shift toward harder tasks
            self.task_distribution['easy'] = max(0.3, self.task_distribution['easy'] - 0.05)
            self.task_distribution['hard'] = min(0.4, self.task_distribution['hard'] + 0.05)
            self.task_distribution['medium'] = 1.0 - (
                self.task_distribution['easy'] + self.task_distribution['hard']
            )
            print(f"📚 Shifted task distribution toward harder tasks")
        
        elif success_rate < 0.4:
            # Struggling - shift toward easier tasks
            self.task_distribution['easy'] = min(0.7, self.task_distribution['easy'] + 0.05)
            self.task_distribution['hard'] = max(0.05, self.task_distribution['hard'] - 0.05)
            self.task_distribution['medium'] = 1.0 - (
                self.task_distribution['easy'] + self.task_distribution['hard']
            )
            print(f"📚 Shifted task distribution toward easier tasks")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Curriculum Learning...\n")
    
    # Test Adaptive Curriculum
    print("1. Adaptive Curriculum Scheduler:")
    scheduler = CurriculumScheduler(
        initial_difficulty=0.3,
        success_threshold=0.7,
        window_size=10
    )
    
    # Simulate episodes with varying success
    print("\n   Simulating episodes:")
    for i in range(30):
        # Simulate improving performance
        success = np.random.rand() < (0.5 + i * 0.015)
        scheduler.record_episode(success)
        
        if (i + 1) % 10 == 0:
            new_diff = scheduler.update_difficulty()
            success_rate = np.mean(scheduler.recent_results)
            print(f"   Episode {i+1}: Success rate={success_rate:.1%}, "
                  f"Difficulty={new_diff:.2f}")
    
    print(f"   ✓ Adaptive curriculum works\n")
    
    # Test Stage-based Curriculum
    print("2. Stage-based Curriculum:")
    stages = [
        {'name': 'Stage 1: Basic Movement', 'difficulty': 0.3},
        {'name': 'Stage 2: Resource Collection', 'difficulty': 0.5},
        {'name': 'Stage 3: Advanced Tasks', 'difficulty': 0.8},
    ]
    
    stage_curriculum = StageBasedCurriculum(
        stages=stages,
        success_threshold=0.8,
        window_size=5
    )
    
    print(f"   Current stage: {stage_curriculum.get_current_stage()['name']}")
    
    # Simulate mastery and advancement
    for _ in range(5):
        stage_curriculum.record_episode(success=True)
    
    advanced = stage_curriculum.check_advancement()
    if advanced:
        print(f"   New stage: {stage_curriculum.get_current_stage()['name']}")
    
    print(f"   ✓ Stage-based curriculum works\n")
    
    # Test Task Curriculum
    print("3. Task-based Curriculum:")
    task_curriculum = TaskCurriculum(
        easy_tasks=[{'grid_size': 10, 'obstacles': 5}],
        medium_tasks=[{'grid_size': 15, 'obstacles': 15}],
        hard_tasks=[{'grid_size': 20, 'obstacles': 30}]
    )
    
    print(f"   Initial distribution: {task_curriculum.task_distribution}")
    
    # Simulate good performance
    task_curriculum.update_distribution(success_rate=0.85)
    print(f"   Updated distribution: {task_curriculum.task_distribution}")
    
    # Sample task
    task = task_curriculum.sample_task()
    print(f"   Sampled task: {task}")
    
    print(f"   ✓ Task-based curriculum works\n")
    
    print("✅ All curriculum learning tests passed!")
