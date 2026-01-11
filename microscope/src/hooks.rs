//! Model Hooks Module
//!
//! This module provides a hook system for intercepting model computations.
//! Hooks can be used to:
//! - Capture activations during forward passes
//! - Modify activations for causal interventions
//! - Monitor model behavior in real-time

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::Result;

/// Type alias for hook functions
pub type HookFn = Arc<dyn Fn(&[f32]) -> Option<Vec<f32>> + Send + Sync>;

/// Represents a registered hook
#[derive(Clone)]
pub struct Hook {
    /// Unique identifier for this hook
    pub name: String,
    /// The hook point (layer, component)
    pub hook_point: String,
    /// The hook function
    pub function: HookFn,
    /// Whether the hook is currently enabled
    pub enabled: bool,
}

impl std::fmt::Debug for Hook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hook")
            .field("name", &self.name)
            .field("hook_point", &self.hook_point)
            .field("enabled", &self.enabled)
            .finish()
    }
}

/// Registry for managing hooks
#[derive(Debug, Default)]
pub struct HookRegistry {
    /// All registered hooks by name
    hooks: HashMap<String, Hook>,
    /// Hooks indexed by hook point for fast lookup
    by_hook_point: HashMap<String, Vec<String>>,
    /// Global enable/disable
    enabled: bool,
}

impl HookRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
            by_hook_point: HashMap::new(),
            enabled: true,
        }
    }

    /// Register a new hook
    pub fn register(
        &mut self,
        name: &str,
        hook_point: &str,
        function: HookFn,
    ) -> Result<()> {
        let hook = Hook {
            name: name.to_string(),
            hook_point: hook_point.to_string(),
            function,
            enabled: true,
        };

        self.hooks.insert(name.to_string(), hook);
        self.by_hook_point
            .entry(hook_point.to_string())
            .or_default()
            .push(name.to_string());

        Ok(())
    }

    /// Remove a hook by name
    pub fn remove(&mut self, name: &str) -> Option<Hook> {
        if let Some(hook) = self.hooks.remove(name) {
            if let Some(hooks) = self.by_hook_point.get_mut(&hook.hook_point) {
                hooks.retain(|n| n != name);
            }
            Some(hook)
        } else {
            None
        }
    }

    /// Get all hooks for a specific hook point
    pub fn get_hooks(&self, hook_point: &str) -> Vec<&Hook> {
        if !self.enabled {
            return vec![];
        }

        self.by_hook_point
            .get(hook_point)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|name| self.hooks.get(name))
                    .filter(|h| h.enabled)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Execute all hooks for a hook point
    pub fn execute(&self, hook_point: &str, data: &[f32]) -> Option<Vec<f32>> {
        let hooks = self.get_hooks(hook_point);
        let mut current_data = None;

        for hook in hooks {
            let input = current_data.as_ref().map(|v: &Vec<f32>| v.as_slice()).unwrap_or(data);
            if let Some(modified) = (hook.function)(input) {
                current_data = Some(modified);
            }
        }

        current_data
    }

    /// Enable/disable a specific hook
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> Result<()> {
        if let Some(hook) = self.hooks.get_mut(name) {
            hook.enabled = enabled;
            Ok(())
        } else {
            Err(crate::MicroscopeError::HookNotFound {
                name: name.to_string(),
            })
        }
    }

    /// Enable/disable all hooks globally
    pub fn set_global_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if hooks are globally enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get all registered hook names
    pub fn hook_names(&self) -> Vec<&str> {
        self.hooks.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all hooks
    pub fn clear(&mut self) {
        self.hooks.clear();
        self.by_hook_point.clear();
    }
}

/// Standard hook points in transformer models
pub mod hook_points {
    /// Embedding layer output
    pub const EMBED: &str = "embed";
    /// Layer norm before attention
    pub const LN1: &str = "ln1";
    /// Attention output (per layer)
    pub fn attn_out(layer: usize) -> String {
        format!("layers.{}.attn_out", layer)
    }
    /// Attention pattern (per layer)
    pub fn attn_pattern(layer: usize) -> String {
        format!("layers.{}.attn_pattern", layer)
    }
    /// MLP output (per layer)
    pub fn mlp_out(layer: usize) -> String {
        format!("layers.{}.mlp_out", layer)
    }
    /// Residual stream (per layer)
    pub fn residual(layer: usize) -> String {
        format!("layers.{}.residual", layer)
    }
    /// Final layer norm
    pub const LN_FINAL: &str = "ln_final";
    /// Unembedding / logits
    pub const UNEMBED: &str = "unembed";
}

/// Builder for creating common hook configurations
pub struct HookBuilder {
    registry: HookRegistry,
}

impl HookBuilder {
    /// Create a new hook builder
    pub fn new() -> Self {
        Self {
            registry: HookRegistry::new(),
        }
    }

    /// Add a capture hook that stores activations
    pub fn capture(mut self, name: &str, hook_point: &str, storage: Arc<RwLock<Vec<f32>>>) -> Self {
        let storage_clone = storage.clone();
        self.registry
            .register(
                name,
                hook_point,
                Arc::new(move |data: &[f32]| {
                    // Handle poisoned lock gracefully
                    let mut guard = match storage_clone.write() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    *guard = data.to_vec();
                    None // Don't modify data
                }),
            )
            .ok();
        self
    }

    /// Add a zero-ablation hook
    pub fn zero_ablate(mut self, name: &str, hook_point: &str) -> Self {
        self.registry
            .register(
                name,
                hook_point,
                Arc::new(|data: &[f32]| Some(vec![0.0; data.len()])),
            )
            .ok();
        self
    }

    /// Add a mean-ablation hook
    pub fn mean_ablate(mut self, name: &str, hook_point: &str, mean_cache: Arc<RwLock<Option<Vec<f32>>>>) -> Self {
        let cache = mean_cache.clone();
        self.registry
            .register(
                name,
                hook_point,
                Arc::new(move |_data: &[f32]| {
                    // Handle poisoned lock gracefully
                    let guard = match cache.read() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    guard.clone()
                }),
            )
            .ok();
        self
    }

    /// Add a patching hook that replaces with cached values
    pub fn patch(mut self, name: &str, hook_point: &str, patch_values: Arc<RwLock<Option<Vec<f32>>>>) -> Self {
        let values = patch_values.clone();
        self.registry
            .register(
                name,
                hook_point,
                Arc::new(move |_data: &[f32]| {
                    // Handle poisoned lock gracefully
                    let guard = match values.read() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    guard.clone()
                }),
            )
            .ok();
        self
    }

    /// Build the registry
    pub fn build(self) -> HookRegistry {
        self.registry
    }
}

impl Default for HookBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_registration() {
        let mut registry = HookRegistry::new();

        registry
            .register(
                "test_hook",
                "layers.0.attn_out",
                Arc::new(|_data| None),
            )
            .unwrap();

        assert!(registry.hooks.contains_key("test_hook"));
        assert_eq!(registry.hook_names().len(), 1);
    }

    #[test]
    fn test_hook_execution() {
        let mut registry = HookRegistry::new();

        // Hook that doubles values
        registry
            .register(
                "double",
                "test_point",
                Arc::new(|data: &[f32]| Some(data.iter().map(|x| x * 2.0).collect())),
            )
            .unwrap();

        let input = vec![1.0, 2.0, 3.0];
        let result = registry.execute("test_point", &input).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_hook_builder() {
        let storage = Arc::new(RwLock::new(Vec::new()));
        let registry = HookBuilder::new()
            .capture("cap", "embed", storage.clone())
            .zero_ablate("zero", "layers.0.attn_out")
            .build();

        assert_eq!(registry.hook_names().len(), 2);
    }

    #[test]
    fn test_hook_disable() {
        let mut registry = HookRegistry::new();

        registry
            .register(
                "test",
                "point",
                Arc::new(|data: &[f32]| Some(data.iter().map(|x| x * 2.0).collect())),
            )
            .unwrap();

        registry.set_enabled("test", false).unwrap();

        let input = vec![1.0, 2.0];
        let result = registry.execute("point", &input);

        assert!(result.is_none()); // Hook disabled, no modification
    }
}
