import psutil
import time
import threading
import tracemalloc
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import gc
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceProfiler:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.process = psutil.Process()
        self.monitoring = False
        self.profile_data = []
        self.tracemalloc_enabled = True
        
        # Memory profiling
        self.memory_snapshots = []
        self.gc_counts = []
        
        logger.info("Resource profiler initialized")
    
    def start_profiling(self, interval: float = 1.0):
        """Start resource profiling"""
        if self.monitoring:
            logger.warning("Profiling already running")
            return
        
        self.monitoring = True
        self.profiling_thread = threading.Thread(
            target=self._profiling_loop,
            args=(interval,)
        )
        self.profiling_thread.daemon = True
        self.profiling_thread.start()
        
        # Start tracemalloc if enabled
        if self.tracemalloc_enabled:
            tracemalloc.start()
        
        logger.info(f"Started resource profiling with {interval}s interval")
    
    def stop_profiling(self):
        """Stop resource profiling"""
        self.monitoring = False
        if hasattr(self, 'profiling_thread'):
            self.profiling_thread.join(timeout=2.0)
        
        # Stop tracemalloc
        if self.tracemalloc_enabled and tracemalloc.is_tracing():
            tracemalloc.stop()
        
        logger.info("Stopped resource profiling")
    
    def _profiling_loop(self, interval: float):
        """Main profiling loop"""
        while self.monitoring:
            try:
                profile_point = self._collect_profile_point()
                self.profile_data.append(profile_point)
                
                # Take memory snapshot periodically
                if self.tracemalloc_enabled and len(self.profile_data) % 10 == 0:
                    snapshot = tracemalloc.take_snapshot()
                    self.memory_snapshots.append({
                        'timestamp': datetime.now().isoformat(),
                        'snapshot': snapshot
                    })
                
                # Record GC stats
                gc_counts = gc.get_count()
                self.gc_counts.append({
                    'timestamp': datetime.now().isoformat(),
                    'gc_counts': gc_counts,
                    'gc_thresholds': gc.get_threshold()
                })
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in profiling loop: {e}")
                time.sleep(interval)
    
    def _collect_profile_point(self) -> Dict[str, Any]:
        """Collect a single profile point"""
        timestamp = datetime.now().isoformat()

        with self.process.oneshot():
            proc_info = {
                'timestamp': timestamp,
                'pid': self.process.pid,
                'cpu_percent': self.process.cpu_percent(),
                'memory_percent': self.process.memory_percent(),
                'memory_info': self.process.memory_info()._asdict(),
                'memory_rss_mb': self.process.memory_info().rss / (1024 * 1024),
                'memory_vms_mb': self.process.memory_info().vms / (1024 * 1024),
                'num_threads': self.process.num_threads(),
                # Handle num_fds safely (platform dependent)
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') and os.name != 'nt' else 0,
                'io_counters': self.process.io_counters()._asdict() if self.process.io_counters() else {},
                'create_time': self.process.create_time(),
                'status': self.process.status()
            }

        sys_info = {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_used': psutil.virtual_memory().used,
            'disk_usage': psutil.disk_usage('/').used,
            # FIX: Handle net_io safely - HAS_NET_IO_COUNTERS often considered always true now
            # Rely on net_io_counters() existing and returning data or None.
            'net_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }

        return {
            'process': proc_info,
            'system': sys_info
        }
    
    def get_profile_data(self) -> List[Dict[str, Any]]:
        """Get collected profile data""" 
        return self.profile_data
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Get memory usage analysis"""
        if not self.profile_data:
            return {}
        
        memory_rss_values = [point['process']['memory_rss_mb'] for point in self.profile_data]
        memory_vms_values = [point['process']['memory_vms_mb'] for point in self.profile_data]
        
        analysis = {
            'memory_rss': {
                'min_mb': min(memory_rss_values) if memory_rss_values else 0,
                'max_mb': max(memory_rss_values) if memory_rss_values else 0,
                'avg_mb': np.mean(memory_rss_values) if memory_rss_values else 0,
                'std_mb': np.std(memory_rss_values) if memory_rss_values else 0
            },
            'memory_vms': {
                'min_mb': min(memory_vms_values) if memory_vms_values else 0,
                'max_mb': max(memory_vms_values) if memory_vms_values else 0,
                'avg_mb': np.mean(memory_vms_values) if memory_vms_values else 0,
                'std_mb': np.std(memory_vms_values) if memory_vms_values else 0
            },
            'total_points': len(self.profile_data),
            'duration_seconds': len(self.profile_data) if self.profile_data else 0  # Assuming 1s intervals
        }
        
        return analysis
    
    def get_cpu_analysis(self) -> Dict[str, Any]:
        """Get CPU usage analysis"""
        if not self.profile_data:
            return {}
        
        cpu_percent_values = [point['process']['cpu_percent'] for point in self.profile_data]
        
        analysis = {
            'cpu_percent': {
                'min': min(cpu_percent_values) if cpu_percent_values else 0,
                'max': max(cpu_percent_values) if cpu_percent_values else 0,
                'avg': np.mean(cpu_percent_values) if cpu_percent_values else 0,
                'std': np.std(cpu_percent_values) if cpu_percent_values else 0
            },
            'total_points': len(self.profile_data)
        }
        
        return analysis
    
    def get_top_memory_consumers(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top memory consumers from tracemalloc snapshots"""
        if not self.memory_snapshots or not self.tracemalloc_enabled:
            return []
        
        try:
            # Get the latest snapshot
            latest_snapshot = self.memory_snapshots[-1]['snapshot']
            
            # Get top memory consumers
            top_stats = latest_snapshot.statistics('lineno')
            
            result = []
            for stat in top_stats[:n]:
                result.append({
                    'filename': stat.traceback.format()[0] if stat.traceback.format() else 'N/A',
                    'size_bytes': stat.size,
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting top memory consumers: {e}")
            return []
    
    def export_profile_data(self, output_dir: str = "./outputs/profiles/"):
        """Export profile data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.profile_data:
            # Flatten profile data for CSV export
            flattened_data = []
            for point in self.profile_data:
                flat_point = {
                    'timestamp': point['process']['timestamp'],
                    'cpu_percent': point['process']['cpu_percent'],
                    'memory_rss_mb': point['process']['memory_rss_mb'],
                    'memory_vms_mb': point['process']['memory_vms_mb'],
                    'num_threads': point['process']['num_threads'],
                    'sys_cpu_percent': point['system']['cpu_percent'],
                    'sys_memory_used': point['system']['memory_used']
                }
                flattened_data.append(flat_point)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(f"{output_dir}/resource_profile.csv", index=False)
        
        # Export memory analysis
        mem_analysis = self.get_memory_analysis()
        with open(f"{output_dir}/memory_analysis.json", 'w') as f:
            import json
            json.dump(mem_analysis, f, indent=2)
        
        # Export CPU analysis
        cpu_analysis = self.get_cpu_analysis()
        with open(f"{output_dir}/cpu_analysis.json", 'w') as f:
            import json
            json.dump(cpu_analysis, f, indent=2)
        
        logger.info(f"Profile data exported to {output_dir}")
    
    def print_summary(self):
        """Print a summary of profiling results"""
        if not self.profile_data:
            print("No profile data available")
            return
        
        mem_analysis = self.get_memory_analysis()
        cpu_analysis = self.get_cpu_analysis()
        
        print("\n=== Resource Profiling Summary ===")
        print(f"Total data points: {mem_analysis['total_points']}")
        print(f"Duration: {mem_analysis['duration_seconds']} seconds")
        print("\nMemory Usage (RSS):")
        print(f"  Min: {mem_analysis['memory_rss']['min_mb']:.2f} MB")
        print(f"  Max: {mem_analysis['memory_rss']['max_mb']:.2f} MB")
        print(f"  Avg: {mem_analysis['memory_rss']['avg_mb']:.2f} MB")
        print(f"  Std: {mem_analysis['memory_rss']['std_mb']:.2f} MB")
        print("\nCPU Usage:")
        print(f"  Min: {cpu_analysis['cpu_percent']['min']:.2f}%")
        print(f"  Max: {cpu_analysis['cpu_percent']['max']:.2f}%")
        print(f"  Avg: {cpu_analysis['cpu_percent']['avg']:.2f}%")
        print(f"  Std: {cpu_analysis['cpu_percent']['std']:.2f}%")

class PerformanceBenchmark:
    def __init__(self):
        self.benchmarks = {}
        logger.info("Performance benchmark initialized")
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark a function execution"""
        import time
        start_time = time.time()
        
        # Profile memory before
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Profile memory after
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        execution_time = time.time() - start_time
        
        benchmark_result = {
            'execution_time': execution_time,
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_delta_mb': mem_after - mem_before,
            'result': result
        }
        
        return benchmark_result
    
    def run_model_inference_benchmark(self, model, X_sample, iterations: int = 100):
        """Benchmark model inference performance"""
        def inference_func():
            return model.predict(X_sample, verbose=0)
        
        results = []
        for i in range(iterations):
            bench_result = self.benchmark_function(inference_func)
            results.append(bench_result)
        
        # Calculate statistics
        times = [r['execution_time'] for r in results]
        memory_deltas = [r['memory_delta_mb'] for r in results]
        
        summary = {
            'iterations': iterations,
            'avg_execution_time': np.mean(times),
            'min_execution_time': min(times),
            'max_execution_time': max(times),
            'std_execution_time': np.std(times),
            'avg_memory_delta_mb': np.mean(memory_deltas),
            'total_time_seconds': sum(times)
        }
        
        self.benchmarks['model_inference'] = summary
        return summary
    
    def get_benchmark_results(self) -> Dict[str, Any]:
        """Get all benchmark results"""
        return self.benchmarks

def test_profiler():
    """Test the resource profiler"""
    profiler = ResourceProfiler()
    
    # Start profiling
    profiler.start_profiling(interval=0.5)
    
    # Simulate some work
    time.sleep(2)
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Print summary
    profiler.print_summary()
    
    # Export data
    profiler.export_profile_data()
    
    print("Profiler test passed!")

if __name__ == "__main__":
    test_profiler()