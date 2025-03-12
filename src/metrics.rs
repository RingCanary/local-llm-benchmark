use serde::Serialize;
use sysinfo::{System, Pid, Disks};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use log::{debug, warn, info};

/// Struct to hold system metrics data
#[derive(Debug, Clone, Serialize)]
pub struct SystemMetrics {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// GPU usage percentage (0-100) if available
    pub gpu_usage: Option<f32>,
    /// Peak memory usage in MB
    pub peak_memory_usage_mb: f32,
}

/// Hardware information
#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    /// CPU model name
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Total system memory in GB
    pub total_memory_gb: f64,
    /// Operating system name and version
    pub os_info: String,
    /// Disk information
    pub disk_info: Vec<DiskInfo>,
    /// GPU information if available
    pub gpu_info: Option<String>,
}

/// Disk information
#[derive(Debug, Clone, Serialize)]
pub struct DiskInfo {
    /// Disk name
    pub name: String,
    /// Total disk space in GB
    pub total_space_gb: f64,
    /// Available disk space in GB
    pub available_space_gb: f64,
}

/// Collector for system metrics during benchmarking
pub struct SystemMetricsCollector {
    #[allow(dead_code)]
    system: System,
    metrics: Arc<Mutex<Vec<SystemMetrics>>>,
    running: Arc<Mutex<bool>>,
    collection_thread: Option<thread::JoinHandle<()>>,
}

impl SystemMetricsCollector {
    /// Create a new SystemMetricsCollector
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        SystemMetricsCollector {
            system,
            metrics: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
            collection_thread: None,
        }
    }
    
    /// Start collecting system metrics
    pub fn start(&mut self) {
        let metrics = Arc::clone(&self.metrics);
        let running = Arc::clone(&self.running);
        
        // Set running flag to true
        {
            let mut is_running = running.lock().unwrap();
            *is_running = true;
        }
        
        // Start collection thread
        self.collection_thread = Some(thread::spawn(move || {
            let mut sys = System::new_all();
            // Convert process ID to the correct type
            let pid = Pid::from(std::process::id() as usize);
            
            while *running.lock().unwrap() {
                sys.refresh_all();
                
                // Get CPU usage
                let cpu_usage = sys.global_cpu_usage();
                
                // Get memory usage
                let mut memory_usage_mb = 0.0;
                let mut process_found = false;
                
                for (process_pid, process) in sys.processes().iter() {
                    if *process_pid == pid {
                        memory_usage_mb = process.memory() as f32 / 1024.0 / 1024.0;
                        process_found = true;
                        break;
                    }
                }
                
                if !process_found {
                    warn!("Could not find current process in system processes");
                }
                
                // Get GPU usage if available (placeholder - would need GPU-specific library)
                let gpu_usage = None;
                
                // Calculate peak memory usage
                let peak_memory_usage_mb = {
                    let metrics_lock = metrics.lock().unwrap();
                    let current_peak = metrics_lock.iter()
                        .map(|m| m.memory_usage_mb)
                        .fold(0.0, f32::max);
                    
                    f32::max(current_peak, memory_usage_mb)
                };
                
                // Store metrics
                {
                    let mut metrics_lock = metrics.lock().unwrap();
                    metrics_lock.push(SystemMetrics {
                        cpu_usage,
                        memory_usage_mb,
                        gpu_usage,
                        peak_memory_usage_mb,
                    });
                }
                
                // Sleep for a short duration before next collection
                thread::sleep(Duration::from_millis(100));
            }
        }));
        
        debug!("Started system metrics collection");
    }
    
    /// Stop collecting system metrics and return the average metrics
    pub fn stop(self) -> SystemMetrics {
        // Set running flag to false
        {
            let mut is_running = self.running.lock().unwrap();
            *is_running = false;
        }
        
        // Wait for collection thread to finish
        if let Some(thread) = self.collection_thread {
            if let Err(e) = thread.join() {
                warn!("Failed to join metrics collection thread: {:?}", e);
            }
        }
        
        // Calculate average metrics
        let metrics_lock = self.metrics.lock().unwrap();
        
        if metrics_lock.is_empty() {
            warn!("No metrics were collected");
            return SystemMetrics {
                cpu_usage: 0.0,
                memory_usage_mb: 0.0,
                gpu_usage: None,
                peak_memory_usage_mb: 0.0,
            };
        }
        
        let avg_cpu_usage = metrics_lock.iter().map(|m| m.cpu_usage).sum::<f32>() / metrics_lock.len() as f32;
        let avg_memory_usage_mb = metrics_lock.iter().map(|m| m.memory_usage_mb).sum::<f32>() / metrics_lock.len() as f32;
        let peak_memory_usage_mb = metrics_lock.iter().map(|m| m.peak_memory_usage_mb).fold(0.0, f32::max);
        
        // For GPU, we need to check if any metrics have GPU data
        let gpu_metrics: Vec<f32> = metrics_lock.iter()
            .filter_map(|m| m.gpu_usage)
            .collect();
        
        let avg_gpu_usage = if !gpu_metrics.is_empty() {
            Some(gpu_metrics.iter().sum::<f32>() / gpu_metrics.len() as f32)
        } else {
            None
        };
        
        debug!("Stopped system metrics collection. Collected {} data points", metrics_lock.len());
        
        SystemMetrics {
            cpu_usage: avg_cpu_usage,
            memory_usage_mb: avg_memory_usage_mb,
            gpu_usage: avg_gpu_usage,
            peak_memory_usage_mb,
        }
    }
}

/// Get hardware information about the system
pub fn get_hardware_info() -> HardwareInfo {
    let mut system = System::new_all();
    system.refresh_all();
    
    // Get CPU information
    let cpu_model = if let Some(cpu) = system.cpus().first() {
        cpu.brand().to_string()
    } else {
        "Unknown CPU".to_string()
    };
    let cpu_cores = system.physical_core_count().unwrap_or(0);
    
    // Get memory information
    let total_memory_gb = system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    
    // Get OS information
    let os_info = format!("{} {}", 
                         System::name().unwrap_or_else(|| "Unknown OS".to_string()), 
                         System::os_version().unwrap_or_else(|| "Unknown version".to_string()));
    
    // Get disk information
    let mut disk_info = Vec::new();
    let disks = Disks::new_with_refreshed_list();
    for disk in disks.list() {
        let total_space_gb = disk.total_space() as f64 / 1024.0 / 1024.0 / 1024.0;
        let available_space_gb = disk.available_space() as f64 / 1024.0 / 1024.0 / 1024.0;
        
        disk_info.push(DiskInfo {
            name: disk.name().to_string_lossy().to_string(),
            total_space_gb,
            available_space_gb,
        });
    }
    
    // Try to get GPU information (this is a placeholder - would need a GPU-specific library)
    let gpu_info = None;
    
    info!("Collected hardware information: {} with {} cores", cpu_model, cpu_cores);
    
    HardwareInfo {
        cpu_model,
        cpu_cores,
        total_memory_gb,
        os_info,
        disk_info,
        gpu_info,
    }
}