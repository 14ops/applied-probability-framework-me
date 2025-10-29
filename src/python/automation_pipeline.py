"""
Full Automation Pipeline Integration

This module provides comprehensive automation pipeline integration for the
Applied Probability Framework, connecting Java backend, Python processing,
and React frontend components for seamless operation.

Features:
- Java-Python bridge for real-time communication
- React frontend integration
- WebSocket communication for live updates
- Database integration for persistent storage
- API endpoints for external access
- Monitoring and logging system
- Error handling and recovery
"""

import asyncio
import json
import logging
import subprocess
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import websockets
import aiohttp
from aiohttp import web
import sqlite3
import os
import signal
import sys

from core.parallel_engine import ParallelSimulationEngine, SimulationConfig
from bayesian_adaptive_model import BayesianAdaptiveModel, create_adaptive_model
from adversarial_gan import AdversarialGAN, create_adversarial_gan
from drl_environment import create_rl_environment
from performance_profiler import PerformanceProfiler, StrategyProfiler


@dataclass
class PipelineConfig:
    """Configuration for the automation pipeline."""
    java_port: int = 8080
    python_port: int = 8081
    react_port: int = 3000
    websocket_port: int = 8082
    database_path: str = "automation_pipeline.db"
    log_level: str = "INFO"
    max_concurrent_simulations: int = 100
    heartbeat_interval: int = 30


@dataclass
class SimulationRequest:
    """Request for simulation execution."""
    request_id: str
    strategy_name: str
    num_simulations: int
    config: Dict[str, Any]
    priority: int = 1
    callback_url: Optional[str] = None


@dataclass
class SimulationResult:
    """Result of simulation execution."""
    request_id: str
    status: str
    results: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    error_message: Optional[str] = None


class DatabaseManager:
    """Database manager for persistent storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simulation requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_requests (
                request_id TEXT PRIMARY KEY,
                strategy_name TEXT,
                num_simulations INTEGER,
                config TEXT,
                priority INTEGER,
                status TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP,
                results TEXT,
                error_message TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp TIMESTAMP
            )
        ''')
        
        # System logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT,
                message TEXT,
                timestamp TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_simulation_request(self, request: SimulationRequest):
        """Save simulation request to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO simulation_requests
            (request_id, strategy_name, num_simulations, config, priority, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.request_id,
            request.strategy_name,
            request.num_simulations,
            json.dumps(request.config),
            request.priority,
            'pending',
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def update_simulation_result(self, result: SimulationResult):
        """Update simulation result in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE simulation_requests
            SET status = ?, completed_at = ?, results = ?, error_message = ?
            WHERE request_id = ?
        ''', (
            result.status,
            result.timestamp,
            json.dumps(result.results),
            result.error_message,
            result.request_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_pending_requests(self) -> List[SimulationRequest]:
        """Get pending simulation requests."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT request_id, strategy_name, num_simulations, config, priority
            FROM simulation_requests
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
        ''')
        
        requests = []
        for row in cursor.fetchall():
            request = SimulationRequest(
                request_id=row[0],
                strategy_name=row[1],
                num_simulations=row[2],
                config=json.loads(row[3]),
                priority=row[4]
            )
            requests.append(request)
        
        conn.close()
        return requests


class JavaBridge:
    """Bridge for communication with Java backend."""
    
    def __init__(self, java_port: int = 8080):
        self.java_port = java_port
        self.java_process = None
        self.is_running = False
    
    def start_java_backend(self):
        """Start Java backend process."""
        try:
            # Start Java backend (assuming it's a Spring Boot application)
            self.java_process = subprocess.Popen([
                'java', '-jar', 'applied-probability-backend.jar',
                '--server.port=' + str(self.java_port)
            ])
            self.is_running = True
            logging.info(f"Java backend started on port {self.java_port}")
        except Exception as e:
            logging.error(f"Failed to start Java backend: {e}")
            raise
    
    def stop_java_backend(self):
        """Stop Java backend process."""
        if self.java_process:
            self.java_process.terminate()
            self.java_process.wait()
            self.is_running = False
            logging.info("Java backend stopped")
    
    async def send_to_java(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data to Java backend."""
        url = f"http://localhost:{self.java_port}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                return await response.json()


class WebSocketManager:
    """WebSocket manager for real-time communication."""
    
    def __init__(self, port: int = 8082):
        self.port = port
        self.clients = set()
        self.server = None
    
    async def register_client(self, websocket, path):
        """Register new WebSocket client."""
        self.clients.add(websocket)
        logging.info(f"New WebSocket client connected: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            logging.info(f"WebSocket client disconnected: {websocket.remote_address}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if self.clients:
            message_str = json.dumps(message)
            disconnected = set()
            
            for client in self.clients:
                try:
                    await client.send(message_str)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected
    
    async def start_server(self):
        """Start WebSocket server."""
        self.server = await websockets.serve(
            self.register_client,
            "localhost",
            self.port
        )
        logging.info(f"WebSocket server started on port {self.port}")
    
    async def stop_server(self):
        """Stop WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logging.info("WebSocket server stopped")


class AutomationPipeline:
    """
    Main automation pipeline coordinator.
    
    This class orchestrates the entire automation pipeline, managing
    communication between Java, Python, and React components.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.is_running = False
        
        # Initialize components
        self.db_manager = DatabaseManager(config.database_path)
        self.java_bridge = JavaBridge(config.java_port)
        self.websocket_manager = WebSocketManager(config.websocket_port)
        
        # Simulation engine
        self.simulation_engine = None
        self.adaptive_model = None
        self.gan_model = None
        
        # Performance monitoring
        self.profiler = PerformanceProfiler()
        self.strategy_profiler = StrategyProfiler(self.profiler)
        
        # Request queue
        self.request_queue = asyncio.Queue()
        self.active_simulations = {}
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('automation_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """Initialize the automation pipeline."""
        self.logger.info("Initializing automation pipeline...")
        
        try:
            # Start Java backend
            self.java_bridge.start_java_backend()
            await asyncio.sleep(5)  # Wait for Java to start
            
            # Start WebSocket server
            await self.websocket_manager.start_server()
            
            # Initialize simulation engine
            sim_config = SimulationConfig()
            self.simulation_engine = ParallelSimulationEngine(sim_config)
            
            # Initialize adaptive model
            strategy_names = ['senku', 'takeshi', 'lelouch', 'kazuya', 'hybrid']
            self.adaptive_model = create_adaptive_model(strategy_names)
            
            # Initialize GAN model
            self.gan_model = create_adversarial_gan()
            
            # Start background tasks
            asyncio.create_task(self._process_request_queue())
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._performance_monitor())
            
            self.is_running = True
            self.logger.info("Automation pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize automation pipeline: {e}")
            raise
    
    async def stop(self):
        """Stop the automation pipeline."""
        self.logger.info("Stopping automation pipeline...")
        
        self.is_running = False
        
        # Stop Java backend
        self.java_bridge.stop_java_backend()
        
        # Stop WebSocket server
        await self.websocket_manager.stop_server()
        
        # Cancel active simulations
        for request_id, task in self.active_simulations.items():
            task.cancel()
        
        self.logger.info("Automation pipeline stopped")
    
    async def submit_simulation_request(self, request: SimulationRequest) -> str:
        """Submit a simulation request."""
        self.logger.info(f"Submitting simulation request: {request.request_id}")
        
        # Save to database
        self.db_manager.save_simulation_request(request)
        
        # Add to queue
        await self.request_queue.put(request)
        
        # Notify via WebSocket
        await self.websocket_manager.broadcast({
            'type': 'simulation_requested',
            'request_id': request.request_id,
            'strategy_name': request.strategy_name,
            'num_simulations': request.num_simulations
        })
        
        return request.request_id
    
    async def _process_request_queue(self):
        """Process simulation requests from the queue."""
        while self.is_running:
            try:
                # Get request from queue
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # Check if we can handle more simulations
                if len(self.active_simulations) >= self.config.max_concurrent_simulations:
                    # Put request back in queue
                    await self.request_queue.put(request)
                    await asyncio.sleep(1)
                    continue
                
                # Start simulation
                task = asyncio.create_task(self._execute_simulation(request))
                self.active_simulations[request.request_id] = task
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing request queue: {e}")
    
    async def _execute_simulation(self, request: SimulationRequest):
        """Execute a simulation request."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing simulation: {request.request_id}")
            
            # Create strategy instance
            strategy = self._create_strategy(request.strategy_name)
            
            # Create simulator
            from optimized_game_simulator import create_optimized_simulator_factory
            simulator_factory = create_optimized_simulator_factory()
            
            # Run simulation
            results = self.simulation_engine.run_batch(
                simulator=simulator_factory(),
                strategy=strategy,
                num_runs=request.num_simulations,
                show_progress=False
            )
            
            # Create result
            execution_time = time.time() - start_time
            result = SimulationResult(
                request_id=request.request_id,
                status='completed',
                results={
                    'mean_reward': results.mean_reward,
                    'std_reward': results.std_reward,
                    'success_rate': results.success_rate,
                    'total_duration': results.total_duration,
                    'num_runs': results.num_runs
                },
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            # Save result
            self.db_manager.update_simulation_result(result)
            
            # Notify via WebSocket
            await self.websocket_manager.broadcast({
                'type': 'simulation_completed',
                'request_id': request.request_id,
                'results': result.results,
                'execution_time': execution_time
            })
            
            # Send to Java backend if callback URL provided
            if request.callback_url:
                await self.java_bridge.send_to_java('simulation/callback', {
                    'request_id': request.request_id,
                    'results': result.results
                })
            
            self.logger.info(f"Simulation completed: {request.request_id}")
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {request.request_id}, error: {e}")
            
            # Create error result
            result = SimulationResult(
                request_id=request.request_id,
                status='failed',
                results={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                error_message=str(e)
            )
            
            # Save error result
            self.db_manager.update_simulation_result(result)
            
            # Notify via WebSocket
            await self.websocket_manager.broadcast({
                'type': 'simulation_failed',
                'request_id': request.request_id,
                'error': str(e)
            })
        
        finally:
            # Remove from active simulations
            if request.request_id in self.active_simulations:
                del self.active_simulations[request.request_id]
    
    def _create_strategy(self, strategy_name: str):
        """Create strategy instance by name."""
        strategy_map = {
            'senku': 'SenkuIshigamiStrategy',
            'takeshi': 'TakeshiKovacsStrategy',
            'lelouch': 'LelouchViBritanniaStrategy',
            'kazuya': 'KazuyaKinoshitaStrategy',
            'hybrid': 'HybridStrategy'
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Import and create strategy
        # This is a simplified version - in practice, you'd have proper imports
        from character_strategies.senku_ishigami import SenkuIshigamiStrategy
        from character_strategies.takeshi_kovacs import TakeshiKovacsStrategy
        from character_strategies.lelouch_vi_britannia import LelouchViBritanniaStrategy
        
        strategy_classes = {
            'senku': SenkuIshigamiStrategy,
            'takeshi': TakeshiKovacsStrategy,
            'lelouch': LelouchViBritanniaStrategy
        }
        
        return strategy_classes[strategy_name]()
    
    async def _heartbeat_monitor(self):
        """Monitor system health and send heartbeats."""
        while self.is_running:
            try:
                # Check Java backend health
                java_healthy = self.java_bridge.is_running
                
                # Check active simulations
                active_count = len(self.active_simulations)
                
                # Send heartbeat
                await self.websocket_manager.broadcast({
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat(),
                    'java_healthy': java_healthy,
                    'active_simulations': active_count,
                    'queue_size': self.request_queue.qsize()
                })
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
    
    async def _performance_monitor(self):
        """Monitor system performance."""
        while self.is_running:
            try:
                # Get performance metrics
                performance_report = self.profiler.get_stats()
                
                # Save to database
                conn = sqlite3.connect(self.config.database_path)
                cursor = conn.cursor()
                
                for func_name, stats in performance_report.items():
                    cursor.execute('''
                        INSERT INTO performance_metrics
                        (strategy_name, metric_name, metric_value, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (func_name, 'avg_duration', stats.avg_duration, datetime.now()))
                
                conn.commit()
                conn.close()
                
                # Notify via WebSocket
                await self.websocket_manager.broadcast({
                    'type': 'performance_update',
                    'metrics': performance_report
                })
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(30)


class APIServer:
    """REST API server for external access."""
    
    def __init__(self, pipeline: AutomationPipeline, port: int = 8081):
        self.pipeline = pipeline
        self.port = port
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        self.app.router.add_post('/api/simulate', self.simulate)
        self.app.router.add_get('/api/status', self.status)
        self.app.router.add_get('/api/results/{request_id}', self.get_results)
        self.app.router.add_get('/api/performance', self.get_performance)
    
    async def simulate(self, request):
        """Handle simulation requests."""
        data = await request.json()
        
        sim_request = SimulationRequest(
            request_id=data['request_id'],
            strategy_name=data['strategy_name'],
            num_simulations=data['num_simulations'],
            config=data.get('config', {}),
            priority=data.get('priority', 1),
            callback_url=data.get('callback_url')
        )
        
        request_id = await self.pipeline.submit_simulation_request(sim_request)
        
        return web.json_response({
            'request_id': request_id,
            'status': 'submitted'
        })
    
    async def status(self, request):
        """Get system status."""
        return web.json_response({
            'is_running': self.pipeline.is_running,
            'active_simulations': len(self.pipeline.active_simulations),
            'queue_size': self.pipeline.request_queue.qsize()
        })
    
    async def get_results(self, request):
        """Get simulation results."""
        request_id = request.match_info['request_id']
        
        # Query database for results
        conn = sqlite3.connect(self.pipeline.config.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, results, error_message, completed_at
            FROM simulation_requests
            WHERE request_id = ?
        ''', (request_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return web.json_response({'error': 'Request not found'}, status=404)
        
        status, results, error_message, completed_at = row
        
        return web.json_response({
            'request_id': request_id,
            'status': status,
            'results': json.loads(results) if results else None,
            'error_message': error_message,
            'completed_at': completed_at
        })
    
    async def get_performance(self, request):
        """Get performance metrics."""
        performance_report = self.pipeline.profiler.get_stats()
        return web.json_response(performance_report)
    
    async def start(self):
        """Start the API server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        logging.info(f"API server started on port {self.port}")


async def main():
    """Main function to run the automation pipeline."""
    config = PipelineConfig()
    pipeline = AutomationPipeline(config)
    api_server = APIServer(pipeline)
    
    try:
        # Initialize pipeline
        await pipeline.initialize()
        
        # Start API server
        await api_server.start()
        
        # Keep running
        while pipeline.is_running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
