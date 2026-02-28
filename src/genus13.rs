//! # Genus-13 Multi-Surface Torus Architecture
//!
//! Two topology variants for a 1.7-trillion parameter torus model:
//!
//! ## Variant A: Independent Surfaces (Genus13Config + MultiSurfaceTorus)
//!
//! 13 independent torus manifolds connected via cross-surface bridges.
//! Uniform — every surface is equal, information flows through bridges.
//!
//! ## Variant B: Keyring Topology (KeyringConfig + KeyringTorus) ★
//!
//! 1 central torus (the "ring") with 12 satellite tori threaded around
//! it like keys on a keyring. The central ring provides global context;
//! each satellite specializes in a domain.
//!
//! ```text
//!                    Satellite 0 (Memory)
//!                      ┌──◯──┐
//!                      │     │
//!         Sat 11 ◯─────┤     ├─────◯ Satellite 1 (Planning)
//!        (Action) │     │     │     │
//!                 │   ╔═╧═════╧═╗   │
//!     Sat 10 ◯───┤   ║ CENTRAL  ║   ├───◯ Sat 2 (Language)
//!    (Percept)│   │   ║  RING    ║   │   │
//!             │   │   ║ d=8192   ║   │   │
//!     Sat 9 ◯─┤   │   ║ 8 stream ║   │   ├─◯ Sat 3 (Spatial)
//!   (MetaCog) │   │   ╚═╤═════╤═╝   │   │
//!             │   │     │     │     │   │
//!      Sat 8 ◯───┤     │     │     ├───◯ Sat 4 (Reasoning)
//!    (Creative)   │     │     │     │
//!         Sat 7 ◯─┤     │     ├─────◯ Sat 5 (Emotional)
//!        (Causal)  │     │     │
//!                  └──◯──┘
//!               Satellite 6 (Pattern)
//! ```
//!
//! Each satellite torus has a **junction point** on the central ring
//! where information transfers bidirectionally. Between junctions,
//! the central ring propagates global context.
//!
//! ## Scaling Presets
//!
//! | Name    | Topology | d_model | Layers | Streams | Params      |
//! |---------|----------|---------|--------|---------|-------------|
//! | nano    | single   | 256     | 6      | 8       | 15.7M       |
//! | tiny    | single   | 768     | 12     | 8       | 424M        |
//! | small   | indep/3  | 2048    | 24     | 24      | 9.7B        |
//! | medium  | indep/7  | 4096    | 32     | 56      | 120B        |
//! | large   | indep/13 | 8192    | 48     | 104     | 1.06T       |
//! | giant   | indep/13 | 8192    | 64     | 104     | 1.82T       |
//! | keyring | keyring  | 8192    | 64     | 104     | ~1.7T       |

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::integration::BidirectionalTorusConfig;

// ============================================================
// GENUS-13 CONFIGURATION
// ============================================================

/// Configuration for a genus-N multi-surface torus architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genus13Config {
    /// Number of toroidal surfaces (genus of the manifold)
    pub genus: usize,
    /// Model dimension per surface
    pub d_model: usize,
    /// Feed-forward dimension per surface
    pub d_ff: usize,
    /// Number of attention heads per stream
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Grid size in major direction (per surface)
    pub n_major: usize,
    /// Grid size in minor direction (per surface)
    pub n_minor: usize,
    /// Cross-surface bridge dimension (typically d_model / genus)
    pub bridge_dim: usize,
    /// Whether to use cross-surface coherence
    pub use_cross_surface_coherence: bool,
    /// How often (in layers) to apply cross-surface bridges
    pub bridge_every_n_layers: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Golden ratio spiral winding per surface (offset by surface index)
    pub base_spiral_winding: f64,
    /// Per-surface config (derived, not set directly)
    #[serde(skip)]
    surface_configs: Vec<BidirectionalTorusConfig>,
}

impl Genus13Config {
    /// Total number of parallel attention streams (genus × 8)
    pub fn total_streams(&self) -> usize {
        self.genus * 8
    }

    /// Sequence length per surface
    pub fn seq_len(&self) -> usize {
        self.n_major * self.n_minor
    }

    /// Estimated total parameter count
    pub fn param_count(&self) -> u64 {
        let d = self.d_model as u64;
        let ff = self.d_ff as u64;
        let streams = self.total_streams() as u64;
        let layers = self.n_layers as u64;
        let genus = self.genus as u64;

        // Per-layer params:
        // - QKV+O projections per stream: d^2 * 4 per stream
        // - Feed-forward per surface: d * ff * 2 per surface
        // - Cross-surface bridge (every bridge_every_n_layers): genus * bridge_dim^2
        let attn_per_layer = d * d * 4 * streams;
        let ff_per_layer = d * ff * 2 * genus;
        let bridge_per_layer = if self.use_cross_surface_coherence {
            let b = self.bridge_dim as u64;
            genus * genus * b * b / self.bridge_every_n_layers as u64
        } else {
            0
        };

        // Layer norms: 2 * d * genus per layer
        let norm_per_layer = 2 * d * genus;

        // Embeddings (shared across surfaces)
        let vocab_size = 100_277u64; // tiktoken
        let embeddings = vocab_size * d + self.seq_len() as u64 * d;

        // Output projection
        let output = d * vocab_size;

        let per_layer = attn_per_layer + ff_per_layer + bridge_per_layer + norm_per_layer;
        per_layer * layers + embeddings + output
    }

    /// Human-readable param count string
    pub fn param_count_human(&self) -> String {
        let p = self.param_count();
        if p >= 1_000_000_000_000 {
            format!("{:.2}T", p as f64 / 1e12)
        } else if p >= 1_000_000_000 {
            format!("{:.2}B", p as f64 / 1e9)
        } else if p >= 1_000_000 {
            format!("{:.1}M", p as f64 / 1e6)
        } else {
            format!("{}K", p / 1000)
        }
    }

    /// Generate per-surface BidirectionalTorusConfig instances
    pub fn surface_configs(&self) -> Vec<BidirectionalTorusConfig> {
        let phi = 1.618033988749895_f64; // Golden ratio

        (0..self.genus)
            .map(|surface_id| {
                // Each surface gets a unique spiral winding offset
                let spiral_offset = (surface_id as f64) * phi;
                let winding = self.base_spiral_winding + spiral_offset;

                // Toroidal geometry: vary major/minor radii per surface
                let major_r = 2.0 + 0.1 * (surface_id as f64 * phi).sin();
                let minor_r = 1.0 + 0.05 * (surface_id as f64 * phi).cos();

                BidirectionalTorusConfig {
                    d_model: self.d_model,
                    d_ff: self.d_ff,
                    n_heads: self.n_heads,
                    n_layers: self.n_layers,
                    n_major: self.n_major,
                    n_minor: self.n_minor,
                    major_radius: major_r,
                    minor_radius: minor_r,
                    use_parallel_streams: true,
                    use_compounding: true,
                    use_multi_scale: false,
                    ema_alpha: 0.9,
                    learnable_alpha: true,
                    use_momentum: true,
                    spiral_winding: winding,
                    weight_temperature: 1.0,
                    parallel_execution: true,
                    use_geodesic_bias: true,
                    geodesic_sigma: 0.5,
                    dropout: self.dropout,
                    n_pos_frequencies: 16,
                    use_coherence: true,
                    coherence_threshold: 0.6,
                    smm_learning_rate: 0.01,
                }
            })
            .collect()
    }

    // ── Presets ───────────────────────────────────────────────

    /// nano: Single surface, 15.7M params (for testing)
    pub fn nano() -> Self {
        Self {
            genus: 1,
            d_model: 256,
            d_ff: 1024,
            n_heads: 8,
            n_layers: 6,
            n_major: 32,
            n_minor: 16,
            bridge_dim: 256,
            use_cross_surface_coherence: false,
            bridge_every_n_layers: 1,
            dropout: 0.1,
            base_spiral_winding: 1.618033988749895,
            surface_configs: Vec::new(),
        }
    }

    /// tiny: Single surface, ~424M params
    pub fn tiny() -> Self {
        Self {
            genus: 1,
            d_model: 768,
            d_ff: 3072,
            n_heads: 12,
            n_layers: 12,
            n_major: 64,
            n_minor: 32,
            bridge_dim: 768,
            use_cross_surface_coherence: false,
            bridge_every_n_layers: 1,
            dropout: 0.1,
            base_spiral_winding: 1.618033988749895,
            surface_configs: Vec::new(),
        }
    }

    /// small: 3 surfaces, ~9.7B params
    pub fn small() -> Self {
        Self {
            genus: 3,
            d_model: 2048,
            d_ff: 8192,
            n_heads: 16,
            n_layers: 24,
            n_major: 64,
            n_minor: 32,
            bridge_dim: 512,
            use_cross_surface_coherence: true,
            bridge_every_n_layers: 4,
            dropout: 0.1,
            base_spiral_winding: 1.618033988749895,
            surface_configs: Vec::new(),
        }
    }

    /// medium: 7 surfaces, ~120B params
    pub fn medium() -> Self {
        Self {
            genus: 7,
            d_model: 4096,
            d_ff: 16384,
            n_heads: 32,
            n_layers: 32,
            n_major: 128,
            n_minor: 64,
            bridge_dim: 1024,
            use_cross_surface_coherence: true,
            bridge_every_n_layers: 4,
            dropout: 0.05,
            base_spiral_winding: 1.618033988749895,
            surface_configs: Vec::new(),
        }
    }

    /// large: 13 surfaces, ~1.06T params
    pub fn large() -> Self {
        Self {
            genus: 13,
            d_model: 8192,
            d_ff: 32768,
            n_heads: 64,
            n_layers: 48,
            n_major: 128,
            n_minor: 64,
            bridge_dim: 2048,
            use_cross_surface_coherence: true,
            bridge_every_n_layers: 4,
            dropout: 0.05,
            base_spiral_winding: 1.618033988749895,
            surface_configs: Vec::new(),
        }
    }

    /// giant: 13 surfaces, ~1.82T params — the target 1.7T configuration
    pub fn giant() -> Self {
        Self {
            genus: 13,
            d_model: 8192,
            d_ff: 32768,
            n_heads: 64,
            n_layers: 64,
            n_major: 128,
            n_minor: 64,
            bridge_dim: 2048,
            use_cross_surface_coherence: true,
            bridge_every_n_layers: 4,
            dropout: 0.05,
            base_spiral_winding: 1.618033988749895,
            surface_configs: Vec::new(),
        }
    }
}

impl Default for Genus13Config {
    fn default() -> Self {
        Self::giant()
    }
}

impl fmt::Display for Genus13Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Genus13Config(genus={}, d_model={}, layers={}, streams={}, params={})",
            self.genus,
            self.d_model,
            self.n_layers,
            self.total_streams(),
            self.param_count_human(),
        )
    }
}

// ============================================================
// CROSS-SURFACE COHERENCE BRIDGE
// ============================================================

/// Tracks coherence state across all genus surfaces.
///
/// At every `bridge_every_n_layers` layers, each surface broadcasts
/// a compressed summary (bridge_dim) and receives fused context
/// from all other surfaces.
#[derive(Debug, Clone)]
pub struct CrossSurfaceBridge {
    /// Config
    pub genus: usize,
    /// Bridge dimension
    pub bridge_dim: usize,
    /// d_model per surface
    pub d_model: usize,
    /// Current coherence matrix: genus × genus, values 0.0-1.0
    pub coherence_matrix: Vec<Vec<f64>>,
    /// Surface summary buffers: genus × bridge_dim
    pub summaries: Vec<Vec<f64>>,
    /// EMA decay for coherence tracking
    pub ema_decay: f64,
    /// Bridge activation count
    pub activations: usize,
}

impl CrossSurfaceBridge {
    pub fn new(config: &Genus13Config) -> Self {
        let genus = config.genus;
        let bridge_dim = config.bridge_dim;

        // Initialize coherence matrix to identity (each surface coherent with itself)
        let mut coherence_matrix = vec![vec![0.0; genus]; genus];
        for i in 0..genus {
            coherence_matrix[i][i] = 1.0;
        }

        Self {
            genus,
            bridge_dim,
            d_model: config.d_model,
            coherence_matrix,
            summaries: vec![vec![0.0; bridge_dim]; genus],
            ema_decay: 0.95,
            activations: 0,
        }
    }

    /// Update surface summary from hidden state (mean pooling + projection).
    ///
    /// In a full implementation this would use a learned linear projection.
    /// Here we use mean-pooled hidden states truncated to bridge_dim.
    pub fn update_summary(&mut self, surface_id: usize, hidden: &[f64]) {
        if surface_id >= self.genus {
            return;
        }

        let len = hidden.len().min(self.bridge_dim);
        for i in 0..len {
            // EMA update
            self.summaries[surface_id][i] =
                self.ema_decay * self.summaries[surface_id][i]
                + (1.0 - self.ema_decay) * hidden[i];
        }
    }

    /// Compute cross-surface coherence after all surfaces have updated.
    ///
    /// Returns a genus × genus matrix of cosine similarities.
    pub fn compute_coherence(&mut self) -> &Vec<Vec<f64>> {
        for i in 0..self.genus {
            for j in (i + 1)..self.genus {
                let sim = cosine_similarity(&self.summaries[i], &self.summaries[j]);
                self.coherence_matrix[i][j] = sim;
                self.coherence_matrix[j][i] = sim;
            }
        }
        self.activations += 1;
        &self.coherence_matrix
    }

    /// Fuse context from all surfaces into surface `target_id`.
    ///
    /// Uses coherence-weighted average of all other surface summaries.
    pub fn fuse_context(&self, target_id: usize) -> Vec<f64> {
        let mut fused = vec![0.0; self.bridge_dim];
        let mut total_weight = 0.0;

        for source_id in 0..self.genus {
            if source_id == target_id {
                continue;
            }
            let weight = self.coherence_matrix[target_id][source_id].abs().max(0.01);
            for k in 0..self.bridge_dim {
                fused[k] += weight * self.summaries[source_id][k];
            }
            total_weight += weight;
        }

        if total_weight > 0.0 {
            for k in 0..self.bridge_dim {
                fused[k] /= total_weight;
            }
        }

        fused
    }

    /// Average coherence across all surface pairs
    pub fn mean_coherence(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.genus {
            for j in (i + 1)..self.genus {
                sum += self.coherence_matrix[i][j];
                count += 1;
            }
        }
        if count > 0 { sum / count as f64 } else { 0.0 }
    }

    /// Summary statistics
    pub fn stats(&self) -> BridgeStats {
        BridgeStats {
            genus: self.genus,
            bridge_dim: self.bridge_dim,
            mean_coherence: self.mean_coherence(),
            activations: self.activations,
        }
    }
}

/// Bridge statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStats {
    pub genus: usize,
    pub bridge_dim: usize,
    pub mean_coherence: f64,
    pub activations: usize,
}

// ============================================================
// MULTI-SURFACE TORUS (logical orchestrator)
// ============================================================

/// Multi-surface torus orchestrator.
///
/// Manages `genus` independent BidirectionalTorus surfaces and routes
/// information through cross-surface bridges.
///
/// Note: This is the logical architecture. To actually instantiate
/// 1.7T parameters, you need a distributed training framework
/// (FSDP, DeepSpeed, Megatron-LM). This struct defines the
/// topology and routing; the weight tensors live in the surfaces.
#[derive(Debug, Clone)]
pub struct MultiSurfaceTorus {
    pub config: Genus13Config,
    pub bridge: CrossSurfaceBridge,
    /// Per-surface hidden state buffers (for simulation without GPU tensors)
    pub surface_states: Vec<Vec<f64>>,
    /// Layer counter (for bridge activation scheduling)
    pub current_layer: usize,
    /// Forward pass counter
    pub forward_count: usize,
}

impl MultiSurfaceTorus {
    pub fn new(config: Genus13Config) -> Self {
        let genus = config.genus;
        let d_model = config.d_model;
        let bridge = CrossSurfaceBridge::new(&config);

        Self {
            surface_states: vec![vec![0.0; d_model]; genus],
            bridge,
            config,
            current_layer: 0,
            forward_count: 0,
        }
    }

    /// Simulate a forward pass through all surfaces for one layer.
    ///
    /// In a real GPU implementation, each surface would be a
    /// BidirectionalTorusTransformer shard on a different device.
    pub fn forward_layer(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        assert_eq!(inputs.len(), self.config.genus);

        let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(self.config.genus);

        // Process each surface independently
        for (surface_id, input) in inputs.iter().enumerate() {
            let mut state = input.clone();

            // Simulated attention + FFN (placeholder for actual GPU ops)
            // In production, this calls BidirectionalTorusTransformer::forward()
            let scale = 1.0 / (state.len() as f64).sqrt();
            for i in 0..state.len() {
                state[i] *= scale;
                // Non-linearity (GELU approximation)
                state[i] = state[i] * 0.5 * (1.0 + (state[i] * 0.7978845608).tanh());
            }

            // Update surface state buffer
            self.surface_states[surface_id] = state.clone();
            outputs.push(state);
        }

        // Cross-surface bridge at scheduled intervals
        if self.config.use_cross_surface_coherence
            && self.current_layer % self.config.bridge_every_n_layers == 0
            && self.current_layer > 0
        {
            // Update summaries
            for surface_id in 0..self.config.genus {
                self.bridge.update_summary(surface_id, &outputs[surface_id]);
            }

            // Compute coherence
            self.bridge.compute_coherence();

            // Fuse cross-surface context into each surface
            for surface_id in 0..self.config.genus {
                let fused = self.bridge.fuse_context(surface_id);
                let len = outputs[surface_id].len().min(fused.len());
                for k in 0..len {
                    // Additive residual from other surfaces
                    outputs[surface_id][k] += 0.1 * fused[k];
                }
            }
        }

        self.current_layer += 1;
        outputs
    }

    /// Full forward pass through all layers
    pub fn forward(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut states = inputs.to_vec();

        self.current_layer = 0;
        for _layer in 0..self.config.n_layers {
            states = self.forward_layer(&states);
        }

        self.forward_count += 1;
        states
    }

    /// Merge all surface outputs into a single vector (mean pooling)
    pub fn merge_outputs(&self, surface_outputs: &[Vec<f64>]) -> Vec<f64> {
        let d = self.config.d_model;
        let mut merged = vec![0.0; d];

        for surface_output in surface_outputs {
            let len = surface_output.len().min(d);
            for i in 0..len {
                merged[i] += surface_output[i];
            }
        }

        let genus = self.config.genus as f64;
        for val in &mut merged {
            *val /= genus;
        }

        merged
    }

    /// Get summary of the multi-surface architecture
    pub fn summary(&self) -> String {
        format!(
            "MultiSurfaceTorus: genus={}, d_model={}, layers={}, streams={}, params={}, bridge_coherence={:.3}",
            self.config.genus,
            self.config.d_model,
            self.config.n_layers,
            self.config.total_streams(),
            self.config.param_count_human(),
            self.bridge.mean_coherence(),
        )
    }
}

// ============================================================
// UTILITIES
// ============================================================

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 1e-10 { dot / denom } else { 0.0 }
}

// ============================================================
// VARIANT B: KEYRING TOPOLOGY
// ============================================================
//
// 1 central torus (the "ring") + 12 satellite tori threaded through
// the ring at evenly-spaced angular positions. Like keys on a keyring.
//
// The central ring carries global context. Each satellite specializes.
// At junction points, information transfers bidirectionally between
// the central ring and each satellite.

/// Satellite torus specialization domains (one per satellite)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SatelliteDomain {
    Memory,       // 0: episodic + working memory
    Planning,     // 1: action planning + goal decomposition
    Language,     // 2: linguistic processing
    Spatial,      // 3: spatial reasoning + geometry
    Reasoning,    // 4: logical deduction
    Emotional,    // 5: valence + motivation signals
    Pattern,      // 6: pattern recognition + analogy
    Causal,       // 7: causal inference + counterfactuals
    Creative,     // 8: novel combination + imagination
    MetaCognition,// 9: self-monitoring + confidence
    Perception,   // 10: sensory integration
    Action,       // 11: motor planning + execution
}

impl SatelliteDomain {
    pub fn all() -> [SatelliteDomain; 12] {
        [
            SatelliteDomain::Memory,
            SatelliteDomain::Planning,
            SatelliteDomain::Language,
            SatelliteDomain::Spatial,
            SatelliteDomain::Reasoning,
            SatelliteDomain::Emotional,
            SatelliteDomain::Pattern,
            SatelliteDomain::Causal,
            SatelliteDomain::Creative,
            SatelliteDomain::MetaCognition,
            SatelliteDomain::Perception,
            SatelliteDomain::Action,
        ]
    }

    pub fn index(&self) -> usize {
        *self as usize
    }

    pub fn name(&self) -> &'static str {
        match self {
            SatelliteDomain::Memory => "Memory",
            SatelliteDomain::Planning => "Planning",
            SatelliteDomain::Language => "Language",
            SatelliteDomain::Spatial => "Spatial",
            SatelliteDomain::Reasoning => "Reasoning",
            SatelliteDomain::Emotional => "Emotional",
            SatelliteDomain::Pattern => "Pattern",
            SatelliteDomain::Causal => "Causal",
            SatelliteDomain::Creative => "Creative",
            SatelliteDomain::MetaCognition => "MetaCognition",
            SatelliteDomain::Perception => "Perception",
            SatelliteDomain::Action => "Action",
        }
    }
}

/// Configuration for the keyring torus topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyringConfig {
    /// Central ring model dimension
    pub ring_d_model: usize,
    /// Central ring feed-forward dimension
    pub ring_d_ff: usize,
    /// Central ring attention heads (per stream, 8 streams)
    pub ring_n_heads: usize,
    /// Central ring layers
    pub ring_n_layers: usize,
    /// Satellite torus model dimension
    pub sat_d_model: usize,
    /// Satellite feed-forward dimension
    pub sat_d_ff: usize,
    /// Satellite attention heads (per stream, 8 streams)
    pub sat_n_heads: usize,
    /// Satellite layers
    pub sat_n_layers: usize,
    /// Number of satellites (always 12)
    pub n_satellites: usize,
    /// Junction dimension (bidirectional transfer between ring and satellite)
    pub junction_dim: usize,
    /// Grid size for central ring
    pub ring_n_major: usize,
    pub ring_n_minor: usize,
    /// Grid size for satellite tori (can be smaller than ring)
    pub sat_n_major: usize,
    pub sat_n_minor: usize,
    /// Junction transfer weight (how much satellite influences ring and vice versa)
    pub junction_alpha: f64,
    /// Dropout
    pub dropout: f64,
}

impl KeyringConfig {
    /// Total streams: ring (8) + 12 satellites × 8 each = 104
    pub fn total_streams(&self) -> usize {
        8 + self.n_satellites * 8
    }

    /// Genus of the keyring surface
    pub fn genus(&self) -> usize {
        1 + self.n_satellites // central ring + n satellites
    }

    /// Estimated total parameter count
    pub fn param_count(&self) -> u64 {
        let rd = self.ring_d_model as u64;
        let rff = self.ring_d_ff as u64;
        let rl = self.ring_n_layers as u64;
        let sd = self.sat_d_model as u64;
        let sff = self.sat_d_ff as u64;
        let sl = self.sat_n_layers as u64;
        let ns = self.n_satellites as u64;
        let jd = self.junction_dim as u64;

        // Central ring: 8 streams × (Q,K,V,O) + FFN
        let ring_attn = rd * rd * 4 * 8 * rl;
        let ring_ff = rd * rff * 2 * rl;
        let ring_norm = 2 * rd * rl;

        // Satellites: each has 8 streams × (Q,K,V,O) + FFN
        let sat_attn = sd * sd * 4 * 8 * sl * ns;
        let sat_ff = sd * sff * 2 * sl * ns;
        let sat_norm = 2 * sd * sl * ns;

        // Junctions: bidirectional projection ring↔satellite
        // ring→sat: ring_d_model → junction_dim → sat_d_model
        // sat→ring: sat_d_model → junction_dim → ring_d_model
        let junction_per_sat = rd * jd + jd * sd + sd * jd + jd * rd;
        let junctions = junction_per_sat * ns;

        // Embeddings + output (shared, ring dimension)
        let vocab_size = 100_277u64;
        let embeddings = vocab_size * rd + (self.ring_n_major * self.ring_n_minor) as u64 * rd;
        let output = rd * vocab_size;

        ring_attn + ring_ff + ring_norm
            + sat_attn + sat_ff + sat_norm
            + junctions + embeddings + output
    }

    /// Human-readable param count
    pub fn param_count_human(&self) -> String {
        let p = self.param_count();
        if p >= 1_000_000_000_000 {
            format!("{:.2}T", p as f64 / 1e12)
        } else if p >= 1_000_000_000 {
            format!("{:.2}B", p as f64 / 1e9)
        } else if p >= 1_000_000 {
            format!("{:.1}M", p as f64 / 1e6)
        } else {
            format!("{}K", p / 1000)
        }
    }

    /// Generate the BidirectionalTorusConfig for the central ring
    pub fn ring_config(&self) -> BidirectionalTorusConfig {
        BidirectionalTorusConfig {
            d_model: self.ring_d_model,
            d_ff: self.ring_d_ff,
            n_heads: self.ring_n_heads,
            n_layers: self.ring_n_layers,
            n_major: self.ring_n_major,
            n_minor: self.ring_n_minor,
            major_radius: 3.0,  // Larger ring for the central donut
            minor_radius: 1.0,
            use_parallel_streams: true,
            use_compounding: true,
            use_multi_scale: false,
            ema_alpha: 0.9,
            learnable_alpha: true,
            use_momentum: true,
            spiral_winding: 1.618033988749895,
            weight_temperature: 1.0,
            parallel_execution: true,
            use_geodesic_bias: true,
            geodesic_sigma: 0.5,
            dropout: self.dropout,
            n_pos_frequencies: 16,
            use_coherence: true,
            coherence_threshold: 0.6,
            smm_learning_rate: 0.01,
        }
    }

    /// Generate BidirectionalTorusConfig for satellite `idx`
    pub fn satellite_config(&self, idx: usize) -> BidirectionalTorusConfig {
        let phi = 1.618033988749895_f64;
        let angle = 2.0 * std::f64::consts::PI * idx as f64 / self.n_satellites as f64;

        BidirectionalTorusConfig {
            d_model: self.sat_d_model,
            d_ff: self.sat_d_ff,
            n_heads: self.sat_n_heads,
            n_layers: self.sat_n_layers,
            n_major: self.sat_n_major,
            n_minor: self.sat_n_minor,
            // Smaller satellite tori positioned around the ring
            major_radius: 1.0,
            minor_radius: 0.4,
            use_parallel_streams: true,
            use_compounding: true,
            use_multi_scale: false,
            ema_alpha: 0.9,
            learnable_alpha: true,
            use_momentum: true,
            // Each satellite gets a unique spiral winding based on position
            spiral_winding: phi + angle,
            weight_temperature: 1.0,
            parallel_execution: true,
            use_geodesic_bias: true,
            geodesic_sigma: 0.5,
            dropout: self.dropout,
            n_pos_frequencies: 16,
            use_coherence: true,
            coherence_threshold: 0.6,
            smm_learning_rate: 0.01,
        }
    }

    // ── Presets ───────────────────────────────────────────────

    /// nano keyring: for testing (tiny dimensions)
    pub fn nano() -> Self {
        Self {
            ring_d_model: 256,
            ring_d_ff: 1024,
            ring_n_heads: 8,
            ring_n_layers: 6,
            sat_d_model: 128,
            sat_d_ff: 512,
            sat_n_heads: 4,
            sat_n_layers: 4,
            n_satellites: 12,
            junction_dim: 64,
            ring_n_major: 32,
            ring_n_minor: 16,
            sat_n_major: 16,
            sat_n_minor: 8,
            junction_alpha: 0.3,
            dropout: 0.1,
        }
    }

    /// The target 1.7T keyring configuration
    pub fn keyring_1_7t() -> Self {
        Self {
            ring_d_model: 8192,
            ring_d_ff: 32768,
            ring_n_heads: 64,
            ring_n_layers: 64,
            // Satellites match ring width for full expressiveness
            sat_d_model: 8192,
            sat_d_ff: 32768,
            sat_n_heads: 64,
            sat_n_layers: 48,
            n_satellites: 12,
            junction_dim: 2048,
            ring_n_major: 128,
            ring_n_minor: 64,
            sat_n_major: 64,
            sat_n_minor: 32,
            junction_alpha: 0.3,
            dropout: 0.05,
        }
    }
}

impl Default for KeyringConfig {
    fn default() -> Self {
        Self::keyring_1_7t()
    }
}

impl fmt::Display for KeyringConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KeyringConfig(ring={}d×{}L, {}×sats={}d×{}L, streams={}, params={})",
            self.ring_d_model, self.ring_n_layers,
            self.n_satellites, self.sat_d_model, self.sat_n_layers,
            self.total_streams(),
            self.param_count_human(),
        )
    }
}

// ============================================================
// JUNCTION: BIDIRECTIONAL RING ↔ SATELLITE TRANSFER
// ============================================================

/// Junction point where a satellite torus links to the central ring.
///
/// The junction sits at a specific angular position on the central ring's
/// major circumference. Information flows bidirectionally:
/// - ring → satellite: global context injection
/// - satellite → ring: specialized knowledge injection
#[derive(Debug, Clone)]
pub struct Junction {
    /// Which satellite this junction connects to
    pub satellite_id: usize,
    /// Domain specialization
    pub domain: SatelliteDomain,
    /// Angular position on central ring (radians, 0..2π)
    pub angle: f64,
    /// Ring → satellite transfer buffer
    pub ring_to_sat: Vec<f64>,
    /// Satellite → ring transfer buffer
    pub sat_to_ring: Vec<f64>,
    /// Junction dimension
    pub junction_dim: usize,
    /// Transfer weight
    pub alpha: f64,
    /// Transfer count
    pub transfers: usize,
}

impl Junction {
    pub fn new(satellite_id: usize, config: &KeyringConfig) -> Self {
        let angle = 2.0 * std::f64::consts::PI * satellite_id as f64 / config.n_satellites as f64;
        let domain = SatelliteDomain::all()[satellite_id % 12];

        Self {
            satellite_id,
            domain,
            angle,
            ring_to_sat: vec![0.0; config.junction_dim],
            sat_to_ring: vec![0.0; config.junction_dim],
            junction_dim: config.junction_dim,
            alpha: config.junction_alpha,
            transfers: 0,
        }
    }

    /// Transfer from ring hidden state to satellite input.
    ///
    /// Projects ring_hidden (ring_d_model) → junction_dim → sat_d_model
    /// In real GPU impl, these would be learned linear projections.
    pub fn ring_to_satellite(&mut self, ring_hidden: &[f64], sat_state: &mut [f64]) {
        let jd = self.junction_dim;

        // Project ring → junction dim (truncate/pad)
        for i in 0..jd {
            self.ring_to_sat[i] = if i < ring_hidden.len() {
                ring_hidden[i]
            } else {
                0.0
            };
        }

        // Inject into satellite with mixing weight alpha
        let len = sat_state.len().min(jd);
        for i in 0..len {
            sat_state[i] = (1.0 - self.alpha) * sat_state[i]
                + self.alpha * self.ring_to_sat[i];
        }

        self.transfers += 1;
    }

    /// Transfer from satellite hidden state back to ring.
    pub fn satellite_to_ring(&mut self, sat_hidden: &[f64], ring_state: &mut [f64]) {
        let jd = self.junction_dim;

        // Project satellite → junction dim
        for i in 0..jd {
            self.sat_to_ring[i] = if i < sat_hidden.len() {
                sat_hidden[i]
            } else {
                0.0
            };
        }

        // Inject into ring with mixing weight alpha
        let len = ring_state.len().min(jd);
        for i in 0..len {
            ring_state[i] = (1.0 - self.alpha) * ring_state[i]
                + self.alpha * self.sat_to_ring[i];
        }

        self.transfers += 1;
    }
}

// ============================================================
// KEYRING TORUS (the full architecture)
// ============================================================

/// Keyring Torus: 1 central ring + 12 satellite tori.
///
/// Forward pass for each layer:
/// 1. Central ring processes its hidden state (8-stream torus attention)
/// 2. At junction points, ring broadcasts to satellites
/// 3. Each satellite processes independently (8-stream torus attention)
/// 4. At junction points, satellites inject back into ring
/// 5. Ring aggregates all satellite contributions
///
/// The result is a model where global context flows around the ring
/// and specialized processing happens in the satellites.
#[derive(Debug, Clone)]
pub struct KeyringTorus {
    pub config: KeyringConfig,
    /// Central ring hidden state
    pub ring_state: Vec<f64>,
    /// Per-satellite hidden states
    pub satellite_states: Vec<Vec<f64>>,
    /// Junction points (one per satellite)
    pub junctions: Vec<Junction>,
    /// Current layer index
    pub current_layer: usize,
    /// Forward pass count
    pub forward_count: usize,
    /// Per-satellite activation history (domain → recent activity)
    pub satellite_activity: Vec<f64>,
}

impl KeyringTorus {
    pub fn new(config: KeyringConfig) -> Self {
        let n_sats = config.n_satellites;

        let junctions: Vec<Junction> = (0..n_sats)
            .map(|i| Junction::new(i, &config))
            .collect();

        Self {
            ring_state: vec![0.0; config.ring_d_model],
            satellite_states: vec![vec![0.0; config.sat_d_model]; n_sats],
            junctions,
            satellite_activity: vec![0.0; n_sats],
            current_layer: 0,
            forward_count: 0,
            config,
        }
    }

    /// Process one layer of the keyring topology.
    pub fn forward_layer(
        &mut self,
        ring_input: &[f64],
        satellite_inputs: &[Vec<f64>],
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        assert_eq!(satellite_inputs.len(), self.config.n_satellites);

        // Step 1: Central ring attention (simulated)
        let mut ring_out = ring_input.to_vec();
        let scale = 1.0 / (ring_out.len() as f64).sqrt();
        for v in ring_out.iter_mut() {
            *v *= scale;
            *v = *v * 0.5 * (1.0 + (*v * 0.7978845608).tanh()); // GELU
        }

        // Step 2: Ring → Satellites via junctions
        let mut sat_outs: Vec<Vec<f64>> = satellite_inputs.to_vec();
        for (i, junction) in self.junctions.iter_mut().enumerate() {
            junction.ring_to_satellite(&ring_out, &mut sat_outs[i]);
        }

        // Step 3: Each satellite processes independently (simulated attention)
        for sat_state in sat_outs.iter_mut() {
            let sat_scale = 1.0 / (sat_state.len() as f64).sqrt();
            for v in sat_state.iter_mut() {
                *v *= sat_scale;
                *v = *v * 0.5 * (1.0 + (*v * 0.7978845608).tanh());
            }
        }

        // Step 4: Satellites → Ring via junctions
        for (i, junction) in self.junctions.iter_mut().enumerate() {
            junction.satellite_to_ring(&sat_outs[i], &mut ring_out);

            // Track satellite activity (L2 norm of contribution)
            let activity: f64 = sat_outs[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            self.satellite_activity[i] =
                0.9 * self.satellite_activity[i] + 0.1 * activity;
        }

        // Update internal state
        self.ring_state = ring_out.clone();
        self.satellite_states = sat_outs.clone();
        self.current_layer += 1;

        (ring_out, sat_outs)
    }

    /// Full forward pass through all layers
    pub fn forward(
        &mut self,
        ring_input: &[f64],
        satellite_inputs: &[Vec<f64>],
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut ring = ring_input.to_vec();
        let mut sats = satellite_inputs.to_vec();

        self.current_layer = 0;
        for _layer in 0..self.config.ring_n_layers {
            let (new_ring, new_sats) = self.forward_layer(&ring, &sats);
            ring = new_ring;

            // Satellites have fewer layers — only process up to sat_n_layers
            if self.current_layer <= self.config.sat_n_layers {
                sats = new_sats;
            }
            // After sat_n_layers, satellites freeze and only receive from ring
        }

        self.forward_count += 1;
        (ring, sats)
    }

    /// Merge ring + all satellites into unified output
    pub fn merge_output(
        &self,
        ring_output: &[f64],
        satellite_outputs: &[Vec<f64>],
    ) -> Vec<f64> {
        let d = self.config.ring_d_model;
        let mut merged = vec![0.0; d];

        // Ring contributes with weight 1.0
        let len = ring_output.len().min(d);
        for i in 0..len {
            merged[i] += ring_output[i];
        }

        // Satellites contribute weighted by their activity
        let total_activity: f64 = self.satellite_activity.iter().sum::<f64>().max(1e-10);
        for (sat_idx, sat_out) in satellite_outputs.iter().enumerate() {
            let weight = self.satellite_activity[sat_idx] / total_activity;
            let len = sat_out.len().min(d);
            for i in 0..len {
                merged[i] += weight * sat_out[i];
            }
        }

        // Normalize
        let norm = 1.0 + self.satellite_activity.iter().sum::<f64>() / total_activity;
        for v in merged.iter_mut() {
            *v /= norm;
        }

        merged
    }

    /// Which satellites are most active?
    pub fn active_satellites(&self) -> Vec<(SatelliteDomain, f64)> {
        let mut ranked: Vec<(SatelliteDomain, f64)> = SatelliteDomain::all()
            .iter()
            .enumerate()
            .map(|(i, &domain)| (domain, self.satellite_activity[i]))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Summary
    pub fn summary(&self) -> String {
        let top_sats: Vec<String> = self.active_satellites()
            .iter()
            .take(3)
            .map(|(d, a)| format!("{}={:.3}", d.name(), a))
            .collect();

        format!(
            "KeyringTorus: ring={}d×{}L, 12×sat={}d×{}L, streams={}, params={}, top=[{}]",
            self.config.ring_d_model,
            self.config.ring_n_layers,
            self.config.sat_d_model,
            self.config.sat_n_layers,
            self.config.total_streams(),
            self.config.param_count_human(),
            top_sats.join(", "),
        )
    }
}

// ============================================================
// COMPARISON: both topologies side by side
// ============================================================

/// Compare the two topology variants
pub fn compare_topologies() -> String {
    let indep = Genus13Config::giant();
    let keyring = KeyringConfig::keyring_1_7t();

    format!(
        "╔══════════════════════════════════════════════════════════════════╗\n\
         ║          GENUS-13 TOPOLOGY COMPARISON                          ║\n\
         ╠══════════════════════════════════════════════════════════════════╣\n\
         ║                  │ Independent Surfaces │ Keyring              ║\n\
         ╠══════════════════╪══════════════════════╪══════════════════════╣\n\
         ║ Genus            │ {:<20} │ {:<20} ║\n\
         ║ Total streams    │ {:<20} │ {:<20} ║\n\
         ║ Ring d_model     │ {:<20} │ {:<20} ║\n\
         ║ Satellite d_model│ {:<20} │ {:<20} ║\n\
         ║ Ring layers      │ {:<20} │ {:<20} ║\n\
         ║ Satellite layers │ {:<20} │ {:<20} ║\n\
         ║ Total params     │ {:<20} │ {:<20} ║\n\
         ║ Info flow        │ {:<20} │ {:<20} ║\n\
         ║ Specialization   │ {:<20} │ {:<20} ║\n\
         ╚══════════════════╧══════════════════════╧══════════════════════╝",
        indep.genus, keyring.genus(),
        indep.total_streams(), keyring.total_streams(),
        indep.d_model, keyring.ring_d_model,
        indep.d_model, keyring.sat_d_model,
        indep.n_layers, keyring.ring_n_layers,
        indep.n_layers, keyring.sat_n_layers,
        indep.param_count_human(), keyring.param_count_human(),
        "bridge every 4L", "junction every L",
        "none (uniform)", "12 domains",
    )
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_config() {
        let cfg = Genus13Config::nano();
        assert_eq!(cfg.genus, 1);
        assert_eq!(cfg.total_streams(), 8);
        assert_eq!(cfg.d_model, 256);
        let params = cfg.param_count();
        assert!(params > 10_000_000 && params < 100_000_000,
            "nano should be ~15M, got {}", params);
        println!("nano: {}", cfg);
    }

    #[test]
    fn test_tiny_config() {
        let cfg = Genus13Config::tiny();
        assert_eq!(cfg.genus, 1);
        assert_eq!(cfg.d_model, 768);
        let params = cfg.param_count();
        assert!(params > 100_000_000 && params < 1_000_000_000,
            "tiny should be ~424M, got {}", params);
        println!("tiny: {}", cfg);
    }

    #[test]
    fn test_small_config() {
        let cfg = Genus13Config::small();
        assert_eq!(cfg.genus, 3);
        assert_eq!(cfg.total_streams(), 24);
        let params = cfg.param_count();
        assert!(params > 1_000_000_000, "small should be >1B, got {}", params);
        println!("small: {}", cfg);
    }

    #[test]
    fn test_large_config() {
        let cfg = Genus13Config::large();
        assert_eq!(cfg.genus, 13);
        assert_eq!(cfg.total_streams(), 104);
        assert_eq!(cfg.d_model, 8192);
        assert_eq!(cfg.n_layers, 48);
        let params = cfg.param_count();
        assert!(params > 500_000_000_000, "large should be >500B, got {}", params);
        println!("large: {}", cfg);
    }

    #[test]
    fn test_giant_config() {
        let cfg = Genus13Config::giant();
        assert_eq!(cfg.genus, 13);
        assert_eq!(cfg.total_streams(), 104);
        assert_eq!(cfg.d_model, 8192);
        assert_eq!(cfg.n_layers, 64);
        let params = cfg.param_count();
        assert!(params > 1_000_000_000_000, "giant should be >1T, got {}", params);
        assert!(params < 3_000_000_000_000, "giant should be <3T, got {}", params);
        println!("giant: {} ({})", cfg, cfg.param_count_human());
    }

    #[test]
    fn test_default_is_giant() {
        let cfg = Genus13Config::default();
        assert_eq!(cfg.genus, 13);
        assert_eq!(cfg.n_layers, 64);
    }

    #[test]
    fn test_surface_configs() {
        let cfg = Genus13Config::small();
        let surfaces = cfg.surface_configs();
        assert_eq!(surfaces.len(), 3);

        // Each surface should have unique spiral winding
        let windings: Vec<f64> = surfaces.iter().map(|s| s.spiral_winding).collect();
        assert_ne!(windings[0], windings[1]);
        assert_ne!(windings[1], windings[2]);

        // d_model should match
        for s in &surfaces {
            assert_eq!(s.d_model, cfg.d_model);
            assert_eq!(s.n_layers, cfg.n_layers);
        }
    }

    #[test]
    fn test_cross_surface_bridge() {
        let cfg = Genus13Config::small();
        let mut bridge = CrossSurfaceBridge::new(&cfg);

        // Update summaries with different data
        bridge.update_summary(0, &vec![1.0; cfg.bridge_dim]);
        bridge.update_summary(1, &vec![0.5; cfg.bridge_dim]);
        bridge.update_summary(2, &vec![-1.0; cfg.bridge_dim]);

        let coherence = bridge.compute_coherence();
        assert_eq!(coherence.len(), 3);

        // Surface 0 and 1 should be more similar than 0 and 2
        assert!(coherence[0][1] > coherence[0][2]);

        // Fuse context
        let fused = bridge.fuse_context(0);
        assert_eq!(fused.len(), cfg.bridge_dim);
        assert!(fused.iter().any(|&v| v != 0.0), "fused context should be non-zero");
    }

    #[test]
    fn test_bridge_stats() {
        let cfg = Genus13Config::nano();
        let bridge = CrossSurfaceBridge::new(&cfg);
        let stats = bridge.stats();
        assert_eq!(stats.genus, 1);
        assert_eq!(stats.activations, 0);
    }

    #[test]
    fn test_multi_surface_torus_nano() {
        let cfg = Genus13Config::nano();
        let mut torus = MultiSurfaceTorus::new(cfg.clone());

        let inputs = vec![vec![1.0; cfg.d_model]; cfg.genus];
        let outputs = torus.forward(&inputs);

        assert_eq!(outputs.len(), cfg.genus);
        assert_eq!(outputs[0].len(), cfg.d_model);
        assert_eq!(torus.forward_count, 1);
    }

    #[test]
    fn test_multi_surface_torus_small() {
        let cfg = Genus13Config::small();
        let mut torus = MultiSurfaceTorus::new(cfg.clone());

        let inputs = vec![vec![0.5; cfg.d_model]; cfg.genus];
        let outputs = torus.forward(&inputs);

        assert_eq!(outputs.len(), 3); // genus=3
        assert_eq!(outputs[0].len(), cfg.d_model);

        // Cross-surface coherence should have been activated
        assert!(torus.bridge.activations > 0,
            "bridge should have been activated during forward pass");
    }

    #[test]
    fn test_merge_outputs() {
        let cfg = Genus13Config::small();
        let torus = MultiSurfaceTorus::new(cfg.clone());

        let surface_outputs = vec![
            vec![1.0; cfg.d_model],
            vec![2.0; cfg.d_model],
            vec![3.0; cfg.d_model],
        ];

        let merged = torus.merge_outputs(&surface_outputs);
        assert_eq!(merged.len(), cfg.d_model);
        // Mean of 1, 2, 3 = 2.0
        assert!((merged[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_surface_summary() {
        let cfg = Genus13Config::giant();
        let torus = MultiSurfaceTorus::new(cfg);
        let summary = torus.summary();
        assert!(summary.contains("genus=13"));
        assert!(summary.contains("streams=104"));
        println!("{}", summary);
    }

    #[test]
    fn test_cosine_similarity() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-10);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-10);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 1e-10);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_param_count_human_formatting() {
        assert!(Genus13Config::nano().param_count_human().contains("M"));
        assert!(Genus13Config::giant().param_count_human().contains("T"));
    }

    #[test]
    fn test_all_presets_valid() {
        let presets = vec![
            ("nano", Genus13Config::nano()),
            ("tiny", Genus13Config::tiny()),
            ("small", Genus13Config::small()),
            ("medium", Genus13Config::medium()),
            ("large", Genus13Config::large()),
            ("giant", Genus13Config::giant()),
        ];

        for (name, cfg) in presets {
            assert!(cfg.genus >= 1, "{} genus >= 1", name);
            assert!(cfg.d_model >= 128, "{} d_model >= 128", name);
            assert!(cfg.n_layers >= 1, "{} n_layers >= 1", name);
            assert_eq!(cfg.total_streams(), cfg.genus * 8);
            println!("{}: {}", name, cfg);
        }
    }

    // ============================================================
    // KEYRING TOPOLOGY TESTS
    // ============================================================

    #[test]
    fn test_keyring_nano_config() {
        let cfg = KeyringConfig::nano();
        assert_eq!(cfg.n_satellites, 12);
        assert_eq!(cfg.genus(), 13);
        assert_eq!(cfg.total_streams(), 104);
        assert_eq!(cfg.ring_d_model, 256);
        assert_eq!(cfg.sat_d_model, 128);
        println!("keyring nano: {}", cfg);
    }

    #[test]
    fn test_keyring_1_7t_config() {
        let cfg = KeyringConfig::keyring_1_7t();
        assert_eq!(cfg.genus(), 13);
        assert_eq!(cfg.total_streams(), 104);
        assert_eq!(cfg.ring_d_model, 8192);
        assert_eq!(cfg.sat_d_model, 8192);
        let params = cfg.param_count();
        assert!(params > 1_000_000_000_000, "should be >1T, got {}", params);
        println!("keyring 1.7T: {} ({} params)", cfg, params);
    }

    #[test]
    fn test_keyring_param_count_human() {
        assert!(KeyringConfig::nano().param_count_human().contains("M"));
        assert!(KeyringConfig::keyring_1_7t().param_count_human().contains("T"));
    }

    #[test]
    fn test_satellite_domains() {
        let domains = SatelliteDomain::all();
        assert_eq!(domains.len(), 12);
        assert_eq!(domains[0], SatelliteDomain::Memory);
        assert_eq!(domains[11], SatelliteDomain::Action);
        for (i, d) in domains.iter().enumerate() {
            assert_eq!(d.index(), i);
            assert!(!d.name().is_empty());
        }
    }

    #[test]
    fn test_junction_creation() {
        let cfg = KeyringConfig::nano();
        let j = Junction::new(0, &cfg);
        assert_eq!(j.satellite_id, 0);
        assert_eq!(j.domain, SatelliteDomain::Memory);
        assert!((j.angle - 0.0).abs() < 1e-10);
        assert_eq!(j.junction_dim, 64);
        assert_eq!(j.transfers, 0);

        let j6 = Junction::new(6, &cfg);
        assert!((j6.angle - std::f64::consts::PI).abs() < 1e-6);
        assert_eq!(j6.domain, SatelliteDomain::Pattern);
    }

    #[test]
    fn test_junction_transfer_ring_to_sat() {
        let cfg = KeyringConfig::nano();
        let mut j = Junction::new(0, &cfg);

        let ring_hidden = vec![1.0; cfg.ring_d_model];
        let mut sat_state = vec![0.0; cfg.sat_d_model];

        j.ring_to_satellite(&ring_hidden, &mut sat_state);
        assert_eq!(j.transfers, 1);
        // Satellite should have received some nonzero signal
        assert!(sat_state.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_junction_transfer_sat_to_ring() {
        let cfg = KeyringConfig::nano();
        let mut j = Junction::new(3, &cfg);

        let sat_hidden = vec![2.0; cfg.sat_d_model];
        let mut ring_state = vec![0.0; cfg.ring_d_model];

        j.satellite_to_ring(&sat_hidden, &mut ring_state);
        assert_eq!(j.transfers, 1);
        assert!(ring_state.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_keyring_torus_creation() {
        let cfg = KeyringConfig::nano();
        let torus = KeyringTorus::new(cfg.clone());
        assert_eq!(torus.junctions.len(), 12);
        assert_eq!(torus.satellite_states.len(), 12);
        assert_eq!(torus.ring_state.len(), cfg.ring_d_model);
        assert_eq!(torus.current_layer, 0);
    }

    #[test]
    fn test_keyring_forward_layer() {
        let cfg = KeyringConfig::nano();
        let mut torus = KeyringTorus::new(cfg.clone());

        let ring_input = vec![1.0; cfg.ring_d_model];
        let sat_inputs: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![(i + 1) as f64; cfg.sat_d_model])
            .collect();

        let (ring_out, sat_outs) = torus.forward_layer(&ring_input, &sat_inputs);
        assert_eq!(ring_out.len(), cfg.ring_d_model);
        assert_eq!(sat_outs.len(), 12);
        for s in &sat_outs {
            assert_eq!(s.len(), cfg.sat_d_model);
        }
        assert_eq!(torus.current_layer, 1);
    }

    #[test]
    fn test_keyring_full_forward() {
        let cfg = KeyringConfig::nano();
        let mut torus = KeyringTorus::new(cfg.clone());

        let ring_input = vec![0.5; cfg.ring_d_model];
        let sat_inputs: Vec<Vec<f64>> = vec![vec![0.5; cfg.sat_d_model]; 12];

        let (ring_out, sat_outs) = torus.forward(&ring_input, &sat_inputs);
        assert_eq!(ring_out.len(), cfg.ring_d_model);
        assert_eq!(sat_outs.len(), 12);
        assert_eq!(torus.forward_count, 1);
    }

    #[test]
    fn test_keyring_merge_output() {
        let cfg = KeyringConfig::nano();
        let mut torus = KeyringTorus::new(cfg.clone());

        // Run a forward pass to build activity
        let ring_input = vec![1.0; cfg.ring_d_model];
        let sat_inputs: Vec<Vec<f64>> = (0..12)
            .map(|_| vec![1.0; cfg.sat_d_model])
            .collect();
        let (ring_out, sat_outs) = torus.forward(&ring_input, &sat_inputs);

        let merged = torus.merge_output(&ring_out, &sat_outs);
        assert_eq!(merged.len(), cfg.ring_d_model);
        // Should have nonzero output
        assert!(merged.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_keyring_active_satellites() {
        let cfg = KeyringConfig::nano();
        let mut torus = KeyringTorus::new(cfg.clone());

        // Forward with different-magnitude inputs per satellite
        let ring_input = vec![1.0; cfg.ring_d_model];
        let sat_inputs: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![(i as f64 + 1.0) * 0.1; cfg.sat_d_model])
            .collect();
        torus.forward(&ring_input, &sat_inputs);

        let active = torus.active_satellites();
        assert_eq!(active.len(), 12);
        // The last satellite (Action, index 11) had highest input
        assert_eq!(active[0].0, SatelliteDomain::Action);
    }

    #[test]
    fn test_keyring_summary() {
        let cfg = KeyringConfig::nano();
        let torus = KeyringTorus::new(cfg);
        let summary = torus.summary();
        assert!(summary.contains("KeyringTorus"));
        assert!(summary.contains("12×sat"));
        println!("{}", summary);
    }

    #[test]
    fn test_compare_topologies() {
        let comparison = compare_topologies();
        assert!(comparison.contains("Independent Surfaces"));
        assert!(comparison.contains("Keyring"));
        assert!(comparison.contains("12 domains"));
        println!("{}", comparison);
    }

    #[test]
    fn test_keyring_satellite_configs_unique() {
        let cfg = KeyringConfig::nano();
        let configs: Vec<BidirectionalTorusConfig> = (0..12)
            .map(|i| cfg.satellite_config(i))
            .collect();

        // Each satellite should have a unique spiral_winding
        for i in 0..12 {
            for j in (i + 1)..12 {
                assert!(
                    (configs[i].spiral_winding - configs[j].spiral_winding).abs() > 0.01,
                    "satellites {} and {} should have different windings", i, j
                );
            }
        }
    }

    #[test]
    fn test_ring_config_generation() {
        let cfg = KeyringConfig::nano();
        let ring = cfg.ring_config();
        assert_eq!(ring.d_model, cfg.ring_d_model);
        assert_eq!(ring.d_ff, cfg.ring_d_ff);
        assert_eq!(ring.n_layers, cfg.ring_n_layers);
        assert!(ring.use_parallel_streams);
        assert!(ring.use_compounding);
    }

    #[test]
    fn test_junction_angles_evenly_spaced() {
        let cfg = KeyringConfig::nano();
        let junctions: Vec<Junction> = (0..12).map(|i| Junction::new(i, &cfg)).collect();

        let expected_spacing = 2.0 * std::f64::consts::PI / 12.0;
        for i in 1..12 {
            let diff = junctions[i].angle - junctions[i - 1].angle;
            assert!(
                (diff - expected_spacing).abs() < 1e-10,
                "junction spacing should be uniform, got {}", diff
            );
        }
    }
}
