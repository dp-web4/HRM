use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineInfo {
    pub pool: String,
    pub lct_id: String,
    #[serde(default)]
    pub mdns_name: String,
    pub gateway_host: String,
    #[serde(default)]
    pub last_seen_ip: String,
    pub gateway_port: u16,
    #[serde(default)]
    pub federation_port: u16,
    #[serde(default)]
    pub model_default: String,
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub hardware: String,
}

#[derive(Debug, Deserialize)]
struct FleetManifest {
    #[serde(default)]
    fleet_version: u32,
    machines: HashMap<String, MachineInfo>,
}

/// Immutable registry of SAGE fleet members loaded from fleet.json.
pub struct FleetRegistry {
    self_machine: String,
    machines: HashMap<String, MachineInfo>,
    version: u32,
}

impl FleetRegistry {
    pub fn load(self_machine: &str, manifest_path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(manifest_path)
            .map_err(|e| format!("failed to read fleet.json: {e}"))?;
        let manifest: FleetManifest = serde_json::from_str(&data)
            .map_err(|e| format!("failed to parse fleet.json: {e}"))?;

        Ok(Self {
            self_machine: self_machine.to_string(),
            machines: manifest.machines,
            version: manifest.fleet_version,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn fleet_size(&self) -> usize {
        self.machines.len()
    }

    pub fn get_all(&self) -> &HashMap<String, MachineInfo> {
        &self.machines
    }

    pub fn get_peers(&self) -> HashMap<&str, &MachineInfo> {
        self.machines
            .iter()
            .filter(|(name, _)| name.as_str() != self.self_machine)
            .map(|(name, info)| (name.as_str(), info))
            .collect()
    }

    pub fn get_peer(&self, name: &str) -> Option<&MachineInfo> {
        self.machines.get(name)
    }

    pub fn get_self_info(&self) -> Option<&MachineInfo> {
        self.machines.get(&self.self_machine)
    }

    pub fn peer_names(&self) -> Vec<&str> {
        self.machines
            .keys()
            .filter(|n| n.as_str() != self.self_machine)
            .map(|n| n.as_str())
            .collect()
    }

    pub fn gateway_url(&self, machine_name: &str) -> Option<String> {
        self.machines.get(machine_name).map(|info| {
            format!("http://{}:{}", info.gateway_host, info.gateway_port)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fleet_path() -> PathBuf {
        PathBuf::from("/home/sprout/ai-workspace/SAGE/sage/federation/fleet.json")
    }

    #[test]
    fn load_real_fleet() {
        let path = fleet_path();
        if !path.exists() {
            return;
        }

        let registry = FleetRegistry::load("sprout", &path).unwrap();
        assert_eq!(registry.fleet_size(), 6);
        assert_eq!(registry.get_peers().len(), 5);
        assert!(registry.get_peer("thor").is_some());
        assert!(registry.get_self_info().is_some());
        assert_eq!(registry.get_self_info().unwrap().gateway_port, 8750);
    }

    #[test]
    fn gateway_url() {
        let path = fleet_path();
        if !path.exists() {
            return;
        }

        let registry = FleetRegistry::load("sprout", &path).unwrap();
        let url = registry.gateway_url("thor").unwrap();
        assert!(url.starts_with("http://"));
        assert!(url.contains("8750"));
    }

    #[test]
    fn peer_names_exclude_self() {
        let path = fleet_path();
        if !path.exists() {
            return;
        }

        let registry = FleetRegistry::load("sprout", &path).unwrap();
        let names = registry.peer_names();
        assert!(!names.contains(&"sprout"));
        assert!(names.contains(&"thor"));
    }
}
