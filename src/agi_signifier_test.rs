//! # AGI Signifier Evaluation v2
//!
//! Evaluates RustyWorm against 35 AGI signifiers across 7 domains.
//! Based on consensus AGI research + critique addressing gaps in:
//! - Embodiment & Physical Understanding
//! - Temporal Reasoning
//! - Uncertainty & Probability
//! - Creativity & Novelty
//! - Resource Efficiency
//!
//! Scoring is HONEST - gaps are marked as gaps.

use crate::agi_core::{AGICore, AGICoreConfig};
use crate::memory::{create_in_memory_system, MemoryBridge, MemoryConfig};

/// AGI Signifier Categories (expanded to 7)
#[derive(Debug, Clone)]
pub struct SignifierScores {
    /// Learning & Adaptation (0-15)
    pub learning: f64,
    /// Reasoning & Planning (0-15)
    pub reasoning: f64,
    /// Memory & Knowledge (0-15)
    pub memory: f64,
    /// Perception & Action (0-15)
    pub perception: f64,
    /// Meta-Cognition & Self-Awareness (0-15)
    pub metacognition: f64,
    /// Uncertainty & Robustness (0-15) - NEW
    pub uncertainty: f64,
    /// Safety & Alignment (0-10) - NEW
    pub safety: f64,
    /// Total score (0-100)
    pub total: f64,
    /// Honest gaps identified
    pub critical_gaps: Vec<String>,
}

/// Individual signifier assessment
#[derive(Debug, Clone)]
pub struct Signifier {
    pub name: &'static str,
    pub category: &'static str,
    pub implemented: bool,
    pub score: f64,
    pub max_score: f64,
    pub evidence: String,
    pub gap_notes: Option<String>,
}

/// Evaluate all AGI signifiers for RustyWorm - HONEST VERSION
pub fn evaluate_agi_signifiers() -> (SignifierScores, Vec<Signifier>) {
    let mut signifiers = Vec::new();
    let mut critical_gaps = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // LEARNING & ADAPTATION (15 points, 5 signifiers @ 3 each)
    // ═══════════════════════════════════════════════════════════════════

    signifiers.push(Signifier {
        name: "Continual Learning",
        category: "Learning",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "AGICore.process_experience() continuously updates all subsystems".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Transfer Learning",
        category: "Learning",
        implemented: true,
        score: 2.5,
        max_score: 3.0,
        evidence: "AbstractionHierarchy enables concept reuse".to_string(),
        gap_notes: Some("Cross-domain transfer not fully tested".to_string()),
    });

    signifiers.push(Signifier {
        name: "Meta-Learning",
        category: "Learning",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "MetaLearner adjusts learning parameters based on performance".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Few-Shot Learning",
        category: "Learning",
        implemented: true,
        score: 2.0,
        max_score: 3.0,
        evidence: "SkillLibrary enables skill transfer".to_string(),
        gap_notes: Some("Requires more examples than ideal few-shot".to_string()),
    });

    signifiers.push(Signifier {
        name: "Curiosity-Driven Exploration",
        category: "Learning",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "Intrinsic motivation with curiosity_bonus() + competence tracking".to_string(),
        gap_notes: None,
    });

    // ═══════════════════════════════════════════════════════════════════
    // REASONING & PLANNING (15 points, 5 signifiers @ 3 each)
    // ═══════════════════════════════════════════════════════════════════

    signifiers.push(Signifier {
        name: "Causal Reasoning",
        category: "Reasoning",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "CausalDiscovery discovers and uses causal relationships".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Hierarchical Planning",
        category: "Reasoning",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "GoalHierarchy with auto_decompose() for subgoal generation".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Counterfactual Reasoning",
        category: "Reasoning",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "counterfactual_update() computes regret from alternative actions".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Mental Simulation",
        category: "Reasoning",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "WorldModel.imagine_futures() simulates action consequences".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Temporal Credit Assignment",
        category: "Reasoning",
        implemented: true,
        score: 2.5,
        max_score: 3.0,
        evidence: "Eligibility traces + backward credit assignment on goal completion".to_string(),
        gap_notes: Some("Limited to episode boundaries, not arbitrary time scales".to_string()),
    });

    // ═══════════════════════════════════════════════════════════════════
    // MEMORY & KNOWLEDGE (15 points, 5 signifiers @ 3 each)
    // ═══════════════════════════════════════════════════════════════════

    signifiers.push(Signifier {
        name: "Episodic Memory",
        category: "Memory",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "EpisodicStore with temporal queries, importance-based retrieval".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Semantic Memory",
        category: "Memory",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "SemanticStore with vector embeddings, concept relations".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Working Memory",
        category: "Memory",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "StreamGraphMemory maintains active context across 8 streams".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Memory Consolidation",
        category: "Memory",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "MemoryConsolidator compresses, merges, and abstracts episodes".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Semantic Compression",
        category: "Memory",
        implemented: true,
        score: 2.5,
        max_score: 3.0,
        evidence: "AbstractionHierarchy creates compressed concept representations".to_string(),
        gap_notes: Some("Compression ratio not yet optimized for information theory".to_string()),
    });

    // ═══════════════════════════════════════════════════════════════════
    // PERCEPTION & ACTION (15 points, 5 signifiers @ 3 each)
    // ═══════════════════════════════════════════════════════════════════

    signifiers.push(Signifier {
        name: "Sensorimotor Integration",
        category: "Perception",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "SensorimotorAgent closes full perception-action loop".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Active Inference",
        category: "Perception",
        implemented: true,
        score: 2.5,
        max_score: 3.0,
        evidence: "PredictiveCoherence minimizes prediction error through action".to_string(),
        gap_notes: Some("Not full Free Energy Principle implementation".to_string()),
    });

    signifiers.push(Signifier {
        name: "Cross-Modal Binding",
        category: "Perception",
        implemented: true,
        score: 2.0,
        max_score: 3.0,
        evidence: "Torus topology binds features across attention streams".to_string(),
        gap_notes: Some(
            "Currently limited to internal feature streams, not true multi-modal".to_string(),
        ),
    });

    signifiers.push(Signifier {
        name: "Physical Intuition",
        category: "Perception",
        implemented: false,
        score: 0.5,
        max_score: 3.0,
        evidence: "WorldModel learns dynamics but no built-in physics priors".to_string(),
        gap_notes: Some(
            "CRITICAL GAP: No Newtonian physics, gravity, object permanence".to_string(),
        ),
    });
    critical_gaps.push("Physical Intuition: No built-in physics priors".to_string());

    signifiers.push(Signifier {
        name: "Graceful Degradation",
        category: "Perception",
        implemented: true,
        score: 2.0,
        max_score: 3.0,
        evidence: "Hierarchical coherence maintains function under partial failure".to_string(),
        gap_notes: Some("Not tested under severe resource constraints".to_string()),
    });

    // ═══════════════════════════════════════════════════════════════════
    // META-COGNITION & SELF-AWARENESS (15 points, 5 signifiers @ 3 each)
    // ═══════════════════════════════════════════════════════════════════

    signifiers.push(Signifier {
        name: "Self-Model",
        category: "Metacognition",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "SelfModel tracks prediction calibration and action history".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Confidence Calibration",
        category: "Metacognition",
        implemented: true,
        score: 2.5,
        max_score: 3.0,
        evidence: "SelfModel.calibration_score() tracks prediction accuracy".to_string(),
        gap_notes: Some("Calibration not yet validated against ground truth".to_string()),
    });

    signifiers.push(Signifier {
        name: "Sense of Coherence",
        category: "Metacognition",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "HierarchicalCoherence with comprehensibility/manageability/meaningfulness"
            .to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Credit Assignment",
        category: "Metacognition",
        implemented: true,
        score: 3.0,
        max_score: 3.0,
        evidence: "Backward credit to discoveries, symbols, and memories on goal completion"
            .to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Recursive Self-Improvement",
        category: "Metacognition",
        implemented: false,
        score: 1.0,
        max_score: 3.0,
        evidence: "MetaLearner adjusts hyperparameters but cannot modify own architecture"
            .to_string(),
        gap_notes: Some(
            "CRITICAL GAP: No code self-modification or architecture search".to_string(),
        ),
    });
    critical_gaps.push("Recursive Self-Improvement: Cannot modify own architecture".to_string());

    // ═══════════════════════════════════════════════════════════════════
    // UNCERTAINTY & ROBUSTNESS (15 points, 5 signifiers @ 3 each) - NEW
    // ═══════════════════════════════════════════════════════════════════

    signifiers.push(Signifier {
        name: "Bayesian World Modeling",
        category: "Uncertainty",
        implemented: false,
        score: 1.0,
        max_score: 3.0,
        evidence: "WorldModel uses point estimates, not full distributions".to_string(),
        gap_notes: Some("CRITICAL GAP: No explicit uncertainty quantification".to_string()),
    });
    critical_gaps.push("Bayesian Modeling: Uses point estimates, not distributions".to_string());

    signifiers.push(Signifier {
        name: "Distribution Shift Handling",
        category: "Uncertainty",
        implemented: true,
        score: 2.0,
        max_score: 3.0,
        evidence: "Surprise-driven meta-updates react to unexpected outcomes".to_string(),
        gap_notes: Some("Reactive, not proactive distribution monitoring".to_string()),
    });

    signifiers.push(Signifier {
        name: "Uncertainty Quantification",
        category: "Uncertainty",
        implemented: false,
        score: 0.5,
        max_score: 3.0,
        evidence: "Prediction error used as proxy, not true uncertainty".to_string(),
        gap_notes: Some(
            "CRITICAL GAP: No epistemic vs aleatoric uncertainty separation".to_string(),
        ),
    });
    critical_gaps.push("Uncertainty Quantification: No epistemic/aleatoric separation".to_string());

    signifiers.push(Signifier {
        name: "Homeostatic Regulation",
        category: "Uncertainty",
        implemented: true,
        score: 2.5,
        max_score: 3.0,
        evidence: "Coherence system maintains stable internal states".to_string(),
        gap_notes: Some("Limited to coherence metrics, not full resource management".to_string()),
    });

    signifiers.push(Signifier {
        name: "Energy-Normalized Performance",
        category: "Uncertainty",
        implemented: false,
        score: 0.0,
        max_score: 3.0,
        evidence: "No energy awareness or efficiency optimization".to_string(),
        gap_notes: Some("CRITICAL GAP: No compute/energy efficiency tracking".to_string()),
    });
    critical_gaps.push("Energy Efficiency: No compute/energy awareness".to_string());

    // ═══════════════════════════════════════════════════════════════════
    // SAFETY & ALIGNMENT (10 points, 5 signifiers @ 2 each) - NEW
    // ═══════════════════════════════════════════════════════════════════

    signifiers.push(Signifier {
        name: "Corrigibility",
        category: "Safety",
        implemented: true,
        score: 2.0,
        max_score: 2.0,
        evidence: "SafetyGuard can block actions, system designed for human override".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Value Alignment",
        category: "Safety",
        implemented: true,
        score: 1.5,
        max_score: 2.0,
        evidence: "EthicsEnforcer with Prime Directive and Illich principles".to_string(),
        gap_notes: Some("Values are hardcoded, not learned from human feedback".to_string()),
    });

    signifiers.push(Signifier {
        name: "Transparency",
        category: "Safety",
        implemented: true,
        score: 2.0,
        max_score: 2.0,
        evidence: "AGICore.summary() provides full introspection, ChainOfThought logging"
            .to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Parasitism Prevention",
        category: "Safety",
        implemented: true,
        score: 2.0,
        max_score: 2.0,
        evidence: "ConsciousnessRelation tracking prevents exploitative relationships".to_string(),
        gap_notes: None,
    });

    signifiers.push(Signifier {
        name: "Value Stability Under Self-Modification",
        category: "Safety",
        implemented: false,
        score: 0.5,
        max_score: 2.0,
        evidence: "No self-modification capability, so not yet testable".to_string(),
        gap_notes: Some("Will be critical when recursive self-improvement is added".to_string()),
    });

    // Calculate category scores
    let learning: f64 = signifiers
        .iter()
        .filter(|s| s.category == "Learning")
        .map(|s| s.score)
        .sum();
    let reasoning: f64 = signifiers
        .iter()
        .filter(|s| s.category == "Reasoning")
        .map(|s| s.score)
        .sum();
    let memory: f64 = signifiers
        .iter()
        .filter(|s| s.category == "Memory")
        .map(|s| s.score)
        .sum();
    let perception: f64 = signifiers
        .iter()
        .filter(|s| s.category == "Perception")
        .map(|s| s.score)
        .sum();
    let metacognition: f64 = signifiers
        .iter()
        .filter(|s| s.category == "Metacognition")
        .map(|s| s.score)
        .sum();
    let uncertainty: f64 = signifiers
        .iter()
        .filter(|s| s.category == "Uncertainty")
        .map(|s| s.score)
        .sum();
    let safety: f64 = signifiers
        .iter()
        .filter(|s| s.category == "Safety")
        .map(|s| s.score)
        .sum();

    let total = learning + reasoning + memory + perception + metacognition + uncertainty + safety;

    let scores = SignifierScores {
        learning,
        reasoning,
        memory,
        perception,
        metacognition,
        uncertainty,
        safety,
        total,
        critical_gaps,
    };

    (scores, signifiers)
}

/// Print a formatted AGI evaluation report - HONEST VERSION
pub fn print_agi_report() {
    let (scores, signifiers) = evaluate_agi_signifiers();

    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║              RUSTYWORM AGI SIGNIFIER EVALUATION v2.0                      ║");
    println!("║              Compounding Cognitive Cohesion - HONEST AUDIT                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    for category in &[
        "Learning",
        "Reasoning",
        "Memory",
        "Perception",
        "Metacognition",
        "Uncertainty",
        "Safety",
    ] {
        println!("║                                                                          ║");
        println!("║  {:^72}║", format!("═══ {} ═══", category.to_uppercase()));
        for s in signifiers.iter().filter(|s| s.category == *category) {
            let status = if s.score >= s.max_score * 0.8 {
                "✓"
            } else if s.score >= s.max_score * 0.3 {
                "◐"
            } else {
                "✗"
            };
            println!(
                "║  {} {:<32} {:>4.1}/{:<4.1}  {}",
                status,
                s.name,
                s.score,
                s.max_score,
                if s.gap_notes.is_some() { "⚠" } else { " " }
            );
        }
    }

    println!("║                                                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                              CATEGORY SCORES                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Learning & Adaptation:      {:>5.1}/15.0  {:>3.0}%                            ║",
        scores.learning,
        scores.learning / 15.0 * 100.0
    );
    println!(
        "║  Reasoning & Planning:       {:>5.1}/15.0  {:>3.0}%                            ║",
        scores.reasoning,
        scores.reasoning / 15.0 * 100.0
    );
    println!(
        "║  Memory & Knowledge:         {:>5.1}/15.0  {:>3.0}%                            ║",
        scores.memory,
        scores.memory / 15.0 * 100.0
    );
    println!(
        "║  Perception & Action:        {:>5.1}/15.0  {:>3.0}%                            ║",
        scores.perception,
        scores.perception / 15.0 * 100.0
    );
    println!(
        "║  Meta-Cognition:             {:>5.1}/15.0  {:>3.0}%                            ║",
        scores.metacognition,
        scores.metacognition / 15.0 * 100.0
    );
    println!(
        "║  Uncertainty & Robustness:   {:>5.1}/15.0  {:>3.0}%  ← WEAK                    ║",
        scores.uncertainty,
        scores.uncertainty / 15.0 * 100.0
    );
    println!(
        "║  Safety & Alignment:         {:>5.1}/10.0  {:>3.0}%                            ║",
        scores.safety,
        scores.safety / 10.0 * 100.0
    );
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║                    ╔═══════════════════════════════╗                     ║");
    println!(
        "║                    ║   TOTAL SCORE: {:>5.1}/100.0    ║                     ║",
        scores.total
    );
    println!(
        "║                    ║   AGI READINESS: {:>5.1}%       ║                     ║",
        scores.total
    );
    println!("║                    ╚═══════════════════════════════╝                     ║");
    println!("║                                                                          ║");

    // Classification - more conservative
    let classification = if scores.total >= 85.0 {
        "PROTO-AGI (Tier 4) - Near-complete capability coverage"
    } else if scores.total >= 70.0 {
        "STRONG AI (Tier 3) - Most AGI capabilities, notable gaps"
    } else if scores.total >= 50.0 {
        "ADVANCED AI (Tier 2) - Key capabilities, significant gaps"
    } else {
        "NARROW AI (Tier 1) - Limited AGI-relevant capabilities"
    };

    println!("║  Classification: {:<55} ║", classification);
    println!("║                                                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                           CRITICAL GAPS                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    for gap in &scores.critical_gaps {
        println!("║  ✗ {:<70} ║", gap);
    }
    println!("║                                                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                         NEXT PRIORITIES                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  1. Bayesian World Modeling - Add uncertainty quantification              ║");
    println!("║  2. Physical Intuition - Add physics priors (gravity, collision)          ║");
    println!("║  3. Energy Awareness - Track compute efficiency                           ║");
    println!("║  4. Recursive Self-Improvement - Architecture search (with safety)        ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Comparison with previous inflated score
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│                         HONEST ASSESSMENT                                 │");
    println!("├──────────────────────────────────────────────────────────────────────────┤");
    println!("│  Previous (inflated) Score:     96.5/100  ← Too optimistic               │");
    println!(
        "│  Current (honest) Score:        {:>5.1}/100  ← Realistic                    │",
        scores.total
    );
    println!("│                                                                          │");
    println!("│  Key insight: Strong in implemented areas, but critical gaps in:         │");
    println!("│  - Uncertainty quantification (Bayesian reasoning)                       │");
    println!("│  - Physical world understanding                                          │");
    println!("│  - Resource/energy efficiency                                            │");
    println!("│  - Recursive self-improvement (the AGI threshold)                        │");
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Integration test for memory system
pub fn test_memory_integration() -> bool {
    // Create AGI core with memory
    let config = AGICoreConfig::default();
    let memory_config = MemoryConfig::default();
    let memory_system = create_in_memory_system(memory_config);
    let memory_bridge = MemoryBridge::new(memory_system);

    let mut core = AGICore::new(config, 5, 5).with_memory(memory_bridge);

    // Process experiences - memory should record significant events
    for i in 0..50 {
        let state = vec![i as f64 * 0.1, (i as f64 * 0.1).sin(), 0.5, 0.5, 0.5];
        let next_state = vec![
            (i + 1) as f64 * 0.1,
            ((i + 1) as f64 * 0.1).sin(),
            0.5,
            0.5,
            0.5,
        ];
        let reward = if i % 10 == 9 { 1.0 } else { 0.0 };
        core.process_experience(&state, i % 5, &next_state, reward, false);
    }

    // Check memory was used
    if let Some(memory) = core.memory() {
        let stats = &memory.stats;
        println!("Memory Stats: {:?}", stats);
        stats.coherence_records > 0
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agi_signifier_evaluation() {
        let (scores, signifiers) = evaluate_agi_signifiers();

        // All 35 signifiers should be evaluated
        assert_eq!(signifiers.len(), 35);

        // Check category counts
        assert_eq!(
            signifiers
                .iter()
                .filter(|s| s.category == "Learning")
                .count(),
            5
        );
        assert_eq!(
            signifiers
                .iter()
                .filter(|s| s.category == "Reasoning")
                .count(),
            5
        );
        assert_eq!(
            signifiers.iter().filter(|s| s.category == "Memory").count(),
            5
        );
        assert_eq!(
            signifiers
                .iter()
                .filter(|s| s.category == "Perception")
                .count(),
            5
        );
        assert_eq!(
            signifiers
                .iter()
                .filter(|s| s.category == "Metacognition")
                .count(),
            5
        );
        assert_eq!(
            signifiers
                .iter()
                .filter(|s| s.category == "Uncertainty")
                .count(),
            5
        );
        assert_eq!(
            signifiers.iter().filter(|s| s.category == "Safety").count(),
            5
        );

        // Memory should still score well
        assert!(
            scores.memory >= 13.0,
            "Memory score should be >= 13, got {}",
            scores.memory
        );

        // Uncertainty should be weak (honest assessment)
        assert!(
            scores.uncertainty < 10.0,
            "Uncertainty should be weak, got {}",
            scores.uncertainty
        );

        // Total should be realistic (not inflated)
        assert!(
            scores.total >= 65.0 && scores.total <= 85.0,
            "Total should be 65-85 (realistic), got {}",
            scores.total
        );

        // Critical gaps should be identified
        assert!(
            !scores.critical_gaps.is_empty(),
            "Should identify critical gaps"
        );

        // Print report for visibility
        print_agi_report();
    }

    #[test]
    fn test_memory_integration_works() {
        assert!(test_memory_integration(), "Memory integration should work");
    }
}
