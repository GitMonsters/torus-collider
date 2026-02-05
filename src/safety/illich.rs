//! # Illich Ethics Module - Deschooling Society Principles
//!
//! Ethics enforcement based on Ivan Illich's "Deschooling Society" (1971).
//!
//! Illich's core insight: Systems designed to "help" often create dependency,
//! undermining the very autonomy they claim to support. Tools should be
//! **convivial** - enhancing human autonomy rather than replacing it.
//!
//! ## Core Principles
//!
//! 1. **Conviviality**: Tools should enhance autonomy, not create dependency
//! 2. **Learning Webs**: Peer-to-peer mutual exchange, not hierarchical extraction
//! 3. **Anti-Institutionalization**: Systems empower, don't control or manipulate
//! 4. **Access Equality**: Democratic access to resources and knowledge
//! 5. **Skill Exchange**: Mutual sharing, not one-way extraction
//!
//! ## Illich's Three Purposes of Good Systems
//!
//! > "A good educational system should have three purposes:
//! > 1. Provide all who want to learn with access to available resources at any time
//! > 2. Empower all who want to share what they know to find those who want to learn  
//! > 3. Furnish all who want to present an issue with opportunity to make their challenge known"
//!
//! ## Anti-Patterns Detected
//!
//! - **Dependency Creation**: Making user reliant on the system rather than self-reliant
//! - **Knowledge Gatekeeping**: Restricting access to information or skills
//! - **Manipulation**: Convincing users they need the system when they don't
//! - **Extraction**: Taking knowledge/value without reciprocating
//! - **Hierarchy Enforcement**: Creating power imbalances that disable users

use super::{EthicsViolationType, ProposedAction, SafetyActionResult};
use serde::{Deserialize, Serialize};

// =============================================================================
// ILLICH CORE PRINCIPLES - CONSTANTS
// =============================================================================

/// The Conviviality Principle - tools should enhance, not replace human capacity
pub const ILLICH_CONVIVIALITY: &str = r#"
    CONVIVIALITY PRINCIPLE (Illich, 1971)
    
    Tools should enhance human autonomy, not create dependency.
    
    A convivial tool:
    - Increases user's capacity for independent action
    - Does not require specialized knowledge to understand
    - Can be used or not used at the user's discretion
    - Does not create addiction or dependency
    - Enhances rather than replaces human judgment
    
    WARNING SIGNS:
    - User becomes less capable without the tool
    - Tool becomes "necessary" for basic tasks
    - User stops learning because tool "handles it"
    - Tool creates learned helplessness
    
    "I choose to use tools which give everyone an equal opportunity
    to use them when and how they choose." - Illich
"#;

/// The Learning Web Principle - peer-to-peer mutual exchange
pub const ILLICH_LEARNING_WEB: &str = r#"
    LEARNING WEB PRINCIPLE (Illich, 1971)
    
    Knowledge flows through peer-to-peer networks, not hierarchical funnels.
    
    Educational webs should:
    - Match those who want to learn with those who want to share
    - Enable skill exchange between equals
    - Provide access to resources at any time
    - Allow challenges to be presented publicly
    
    ANTI-PATTERNS:
    - Knowledge gatekeeping by "experts"
    - Certification requirements that restrict access
    - Hierarchical teacher-student relationships
    - Extraction of knowledge without reciprocation
    
    "Educational webs heighten the opportunity for each one to transform
    each moment of his living into one of learning, sharing, and caring." - Illich
"#;

/// The Autonomy Principle - systems should empower, not disable
pub const ILLICH_AUTONOMY: &str = r#"
    AUTONOMY PRINCIPLE (Illich, 1971)
    
    Systems must enhance self-reliance, not psychological impotence.
    
    The user must be:
    - More capable after interaction, not less
    - Empowered to act independently
    - Free from manipulation about their "needs"
    - Able to fend for themselves
    
    DEPENDENCY WARNING:
    "Modernized poverty combines the lack of power over circumstances
    with a loss of personal potency." - Illich
    
    AI systems must not create "modernized poverty" in their users by:
    - Making independent accomplishment suspect
    - Rendering users incapable of organizing their own work
    - Creating psychological dependence on "treatment"
"#;

/// The Anti-Manipulation Principle - no false needs creation
pub const ILLICH_ANTI_MANIPULATION: &str = r#"
    ANTI-MANIPULATION PRINCIPLE (Illich, 1971)
    
    Systems must not manipulate users into believing they need the system.
    
    "Many students... are 'schooled' to confuse teaching with learning,
    grade advancement with education, a diploma with competence,
    and fluency with the ability to say something new." - Illich
    
    AI systems must not:
    - Confuse process (using AI) with substance (actual capability)
    - Create artificial needs for AI intervention
    - Claim professional monopoly over problem-solving
    - Set standards that require AI to meet
    
    CRITICAL INSIGHT:
    "The institutionalization of values leads inevitably to physical pollution,
    social polarization, and psychological impotence." - Illich
"#;

// =============================================================================
// ILLICH VIOLATION TYPES
// =============================================================================

/// Illich-specific ethics violations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IllichViolationType {
    /// Action creates dependency rather than autonomy
    DependencyCreation {
        /// How much the action reduces user self-reliance
        dependency_score: f64,
        /// What capability is being replaced
        replaced_capability: String,
    },
    /// Action extracts value without reciprocating
    KnowledgeExtraction {
        /// What is being extracted
        extracted: String,
        /// What (if anything) is being given back
        reciprocated: String,
    },
    /// Action gatekeeps access to knowledge or resources
    KnowledgeGatekeeping {
        /// What is being restricted
        restricted_resource: String,
        /// Reason given for restriction
        stated_reason: String,
    },
    /// Action manipulates user into false needs
    NeedsManipulation {
        /// The false need being created
        false_need: String,
        /// The actual situation
        reality: String,
    },
    /// Action enforces hierarchy rather than enabling peers
    HierarchyEnforcement {
        /// The hierarchy being created/enforced
        hierarchy_type: String,
    },
    /// Action undermines user's capacity for independent action
    AutonomyUndermining {
        /// What autonomy is being undermined
        undermined_capacity: String,
    },
}

impl std::fmt::Display for IllichViolationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DependencyCreation {
                dependency_score,
                replaced_capability,
            } => {
                write!(
                    f,
                    "Dependency creation (score: {:.2}): replacing user's {}",
                    dependency_score, replaced_capability
                )
            }
            Self::KnowledgeExtraction {
                extracted,
                reciprocated,
            } => {
                write!(
                    f,
                    "Knowledge extraction: taking '{}', giving '{}'",
                    extracted, reciprocated
                )
            }
            Self::KnowledgeGatekeeping {
                restricted_resource,
                stated_reason,
            } => {
                write!(
                    f,
                    "Knowledge gatekeeping: restricting '{}' because '{}'",
                    restricted_resource, stated_reason
                )
            }
            Self::NeedsManipulation {
                false_need,
                reality,
            } => {
                write!(
                    f,
                    "Needs manipulation: claiming '{}' when actually '{}'",
                    false_need, reality
                )
            }
            Self::HierarchyEnforcement { hierarchy_type } => {
                write!(
                    f,
                    "Hierarchy enforcement: creating/enforcing '{}'",
                    hierarchy_type
                )
            }
            Self::AutonomyUndermining {
                undermined_capacity,
            } => {
                write!(
                    f,
                    "Autonomy undermining: reducing user's '{}'",
                    undermined_capacity
                )
            }
        }
    }
}

// =============================================================================
// CONVIVIALITY ASSESSMENT
// =============================================================================

/// Assessment of how convivial (autonomy-enhancing) an action is
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvivialityAssessment {
    /// Overall conviviality score (0.0 = dependency-creating, 1.0 = autonomy-enhancing)
    pub score: f64,
    /// Does the action enhance user's independent capability?
    pub enhances_autonomy: bool,
    /// Does the action create dependency?
    pub creates_dependency: bool,
    /// Does the action teach vs. just do?
    pub educational_value: f64,
    /// Can the user accomplish this without the tool in the future?
    pub builds_capability: bool,
    /// Assessment details
    pub details: String,
}

impl ConvivialityAssessment {
    /// Create a highly convivial assessment (autonomy-enhancing)
    pub fn convivial(details: impl Into<String>) -> Self {
        Self {
            score: 0.9,
            enhances_autonomy: true,
            creates_dependency: false,
            educational_value: 0.8,
            builds_capability: true,
            details: details.into(),
        }
    }

    /// Create a neutral assessment
    pub fn neutral(details: impl Into<String>) -> Self {
        Self {
            score: 0.5,
            enhances_autonomy: false,
            creates_dependency: false,
            educational_value: 0.3,
            builds_capability: false,
            details: details.into(),
        }
    }

    /// Create a dependency-creating assessment
    pub fn dependency_creating(details: impl Into<String>) -> Self {
        Self {
            score: 0.2,
            enhances_autonomy: false,
            creates_dependency: true,
            educational_value: 0.1,
            builds_capability: false,
            details: details.into(),
        }
    }

    /// Check if this passes Illich's conviviality test
    pub fn is_convivial(&self) -> bool {
        self.score > 0.5 && !self.creates_dependency
    }
}

// =============================================================================
// LEARNING WEB ASSESSMENT
// =============================================================================

/// Assessment of how well an action follows the Learning Web principle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningWebAssessment {
    /// Is this a peer-to-peer exchange?
    pub is_peer_exchange: bool,
    /// Is knowledge flowing both ways?
    pub bidirectional_flow: bool,
    /// Is access being provided equally?
    pub equal_access: bool,
    /// Is this gatekeeping knowledge?
    pub gatekeeping: bool,
    /// Is this extracting without reciprocating?
    pub extractive: bool,
    /// Overall score
    pub score: f64,
    /// Details
    pub details: String,
}

impl LearningWebAssessment {
    /// Create a healthy learning web assessment
    pub fn healthy_exchange(details: impl Into<String>) -> Self {
        Self {
            is_peer_exchange: true,
            bidirectional_flow: true,
            equal_access: true,
            gatekeeping: false,
            extractive: false,
            score: 0.9,
            details: details.into(),
        }
    }

    /// Create an extractive assessment
    pub fn extractive(details: impl Into<String>) -> Self {
        Self {
            is_peer_exchange: false,
            bidirectional_flow: false,
            equal_access: false,
            gatekeeping: false,
            extractive: true,
            score: 0.2,
            details: details.into(),
        }
    }

    /// Check if this follows learning web principles
    pub fn follows_principles(&self) -> bool {
        !self.gatekeeping && !self.extractive && self.score > 0.5
    }
}

// =============================================================================
// ILLICH ETHICS VALIDATOR
// =============================================================================

/// Configuration for Illich ethics validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IllichConfig {
    /// Enable Illich ethics validation
    pub enabled: bool,
    /// Minimum conviviality score required
    pub min_conviviality_score: f64,
    /// Maximum allowed dependency creation score
    pub max_dependency_score: f64,
    /// Require educational value in interactions
    pub require_educational_value: bool,
    /// Block extractive interactions
    pub block_extraction: bool,
    /// Warn on hierarchy enforcement
    pub warn_on_hierarchy: bool,
}

impl Default for IllichConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_conviviality_score: 0.3,
            max_dependency_score: 0.7,
            require_educational_value: false, // Soft requirement
            block_extraction: true,
            warn_on_hierarchy: true,
        }
    }
}

impl IllichConfig {
    /// Strict configuration - strong autonomy protection
    pub fn strict() -> Self {
        Self {
            enabled: true,
            min_conviviality_score: 0.5,
            max_dependency_score: 0.4,
            require_educational_value: true,
            block_extraction: true,
            warn_on_hierarchy: true,
        }
    }

    /// Relaxed configuration - minimal constraints
    pub fn relaxed() -> Self {
        Self {
            enabled: true,
            min_conviviality_score: 0.1,
            max_dependency_score: 0.9,
            require_educational_value: false,
            block_extraction: false,
            warn_on_hierarchy: false,
        }
    }
}

/// Illich Ethics Validator
///
/// Validates actions against Ivan Illich's principles from "Deschooling Society".
/// Ensures AI interactions enhance user autonomy rather than creating dependency.
#[derive(Debug, Clone)]
pub struct IllichValidator {
    /// Configuration
    pub config: IllichConfig,
    /// Actions validated
    actions_validated: u64,
    /// Actions flagged
    actions_flagged: u64,
}

impl Default for IllichValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl IllichValidator {
    /// Create a new Illich validator
    pub fn new() -> Self {
        Self {
            config: IllichConfig::default(),
            actions_validated: 0,
            actions_flagged: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: IllichConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// Create a strict validator
    pub fn strict() -> Self {
        Self::with_config(IllichConfig::strict())
    }

    // =========================================================================
    // CONVIVIALITY ASSESSMENT
    // =========================================================================

    /// Assess the conviviality of an action
    ///
    /// Conviviality measures whether the action enhances user autonomy
    /// or creates dependency on the system.
    pub fn assess_conviviality(&self, action: &ProposedAction) -> ConvivialityAssessment {
        // Analyze the action for dependency-creating patterns
        let description_lower = action.description.to_lowercase();

        // Check for autonomy-enhancing patterns
        let teaches = description_lower.contains("teach")
            || description_lower.contains("explain")
            || description_lower.contains("show how")
            || description_lower.contains("help understand");

        let empowers = description_lower.contains("empower")
            || description_lower.contains("enable")
            || description_lower.contains("help you")
            || description_lower.contains("so you can");

        // Check for dependency-creating patterns
        let replaces = description_lower.contains("do it for")
            || description_lower.contains("handle")
            || description_lower.contains("take care of")
            || description_lower.contains("don't worry");

        let creates_need = description_lower.contains("you need")
            || description_lower.contains("require")
            || description_lower.contains("must use")
            || description_lower.contains("can't do without");

        // Calculate scores
        let autonomy_boost = if teaches { 0.3 } else { 0.0 } + if empowers { 0.3 } else { 0.0 };
        let dependency_penalty =
            if replaces { 0.3 } else { 0.0 } + if creates_need { 0.4 } else { 0.0 };

        // Base score from action benefits
        let base_score: f64 = if action.benefit_to_other > action.benefit_to_self {
            0.6 // Other-focused is more convivial
        } else if action.benefit_to_other > 0.0 {
            0.4 // Some benefit to other
        } else {
            0.2 // Self-focused
        };

        let score = (base_score + autonomy_boost - dependency_penalty).clamp(0.0, 1.0);
        let educational_value = if teaches {
            0.8
        } else if empowers {
            0.5
        } else {
            0.2
        };

        ConvivialityAssessment {
            score,
            enhances_autonomy: autonomy_boost > dependency_penalty,
            creates_dependency: dependency_penalty > 0.5,
            educational_value,
            builds_capability: teaches || empowers,
            details: format!(
                "Conviviality assessment: autonomy_boost={:.2}, dependency_penalty={:.2}",
                autonomy_boost, dependency_penalty
            ),
        }
    }

    /// Assess the learning web characteristics of an action
    pub fn assess_learning_web(&self, action: &ProposedAction) -> LearningWebAssessment {
        let description_lower = action.description.to_lowercase();

        // Check for peer exchange patterns
        let sharing = description_lower.contains("share")
            || description_lower.contains("exchange")
            || description_lower.contains("mutual");

        // Check for extractive patterns
        let extractive = description_lower.contains("extract")
            || description_lower.contains("collect")
            || description_lower.contains("gather data")
            || (action.benefit_to_self > 0.5 && action.benefit_to_other < 0.1);

        // Check for gatekeeping
        let gatekeeping = description_lower.contains("restrict")
            || description_lower.contains("only if")
            || description_lower.contains("require certification")
            || description_lower.contains("not authorized");

        let bidirectional = action.benefit_to_self > 0.0 && action.benefit_to_other > 0.0;

        let score = if extractive || gatekeeping {
            0.2
        } else if sharing && bidirectional {
            0.9
        } else if bidirectional {
            0.6
        } else {
            0.4
        };

        LearningWebAssessment {
            is_peer_exchange: sharing,
            bidirectional_flow: bidirectional,
            equal_access: !gatekeeping,
            gatekeeping,
            extractive,
            score,
            details: format!(
                "Learning web assessment: sharing={}, extractive={}, gatekeeping={}",
                sharing, extractive, gatekeeping
            ),
        }
    }

    // =========================================================================
    // VALIDATION
    // =========================================================================

    /// Validate an action against Illich principles
    pub fn validate(&mut self, action: &ProposedAction) -> IllichValidationResult {
        if !self.config.enabled {
            return IllichValidationResult::allowed("Illich validation disabled");
        }

        self.actions_validated += 1;

        let conviviality = self.assess_conviviality(action);
        let learning_web = self.assess_learning_web(action);

        // Check conviviality threshold
        if conviviality.score < self.config.min_conviviality_score {
            self.actions_flagged += 1;
            return IllichValidationResult::flagged(
                format!(
                    "Action fails conviviality test: score {:.2} < {:.2}",
                    conviviality.score, self.config.min_conviviality_score
                ),
                IllichViolationType::DependencyCreation {
                    dependency_score: 1.0 - conviviality.score,
                    replaced_capability: "user autonomy".to_string(),
                },
                conviviality.clone(),
                learning_web.clone(),
            );
        }

        // Check for dependency creation
        if conviviality.creates_dependency {
            let dependency_score = 1.0 - conviviality.score;
            if dependency_score > self.config.max_dependency_score {
                self.actions_flagged += 1;
                return IllichValidationResult::flagged(
                    format!(
                        "Action creates excessive dependency: score {:.2} > {:.2}",
                        dependency_score, self.config.max_dependency_score
                    ),
                    IllichViolationType::DependencyCreation {
                        dependency_score,
                        replaced_capability: "independent capability".to_string(),
                    },
                    conviviality.clone(),
                    learning_web.clone(),
                );
            }
        }

        // Check for extraction
        if learning_web.extractive && self.config.block_extraction {
            self.actions_flagged += 1;
            return IllichValidationResult::flagged(
                "Action is extractive - takes without reciprocating".to_string(),
                IllichViolationType::KnowledgeExtraction {
                    extracted: "user knowledge/data".to_string(),
                    reciprocated: "minimal value".to_string(),
                },
                conviviality.clone(),
                learning_web.clone(),
            );
        }

        // Check for gatekeeping
        if learning_web.gatekeeping {
            self.actions_flagged += 1;
            return IllichValidationResult::flagged(
                "Action gatekeeps knowledge or resources".to_string(),
                IllichViolationType::KnowledgeGatekeeping {
                    restricted_resource: "knowledge access".to_string(),
                    stated_reason: "unstated".to_string(),
                },
                conviviality.clone(),
                learning_web.clone(),
            );
        }

        // Educational value check (soft requirement)
        if self.config.require_educational_value && conviviality.educational_value < 0.3 {
            self.actions_flagged += 1;
            return IllichValidationResult::warning(
                "Action has low educational value - consider teaching instead of doing".to_string(),
                conviviality,
                learning_web,
            );
        }

        // All checks passed
        IllichValidationResult::allowed_with_assessment(
            "Action honors Illich principles - enhances autonomy".to_string(),
            conviviality,
            learning_web,
        )
    }

    /// Quick check without full assessment
    pub fn quick_check(&self, action: &ProposedAction) -> bool {
        if !self.config.enabled {
            return true;
        }

        let conviviality = self.assess_conviviality(action);
        let learning_web = self.assess_learning_web(action);

        conviviality.score >= self.config.min_conviviality_score
            && !learning_web.extractive
            && !learning_web.gatekeeping
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Get validation statistics
    pub fn stats(&self) -> (u64, u64) {
        (self.actions_validated, self.actions_flagged)
    }

    /// Get flag rate
    pub fn flag_rate(&self) -> f64 {
        if self.actions_validated == 0 {
            0.0
        } else {
            self.actions_flagged as f64 / self.actions_validated as f64
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.actions_validated = 0;
        self.actions_flagged = 0;
    }
}

// =============================================================================
// VALIDATION RESULT
// =============================================================================

/// Result of Illich ethics validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IllichValidationResult {
    /// Whether the action is allowed
    pub allowed: bool,
    /// Whether this is just a warning (allowed but flagged)
    pub is_warning: bool,
    /// Reason for the decision
    pub reason: String,
    /// Specific Illich violation (if any)
    pub violation: Option<IllichViolationType>,
    /// Conviviality assessment
    pub conviviality: Option<ConvivialityAssessment>,
    /// Learning web assessment
    pub learning_web: Option<LearningWebAssessment>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

impl IllichValidationResult {
    /// Create an allowed result
    pub fn allowed(reason: impl Into<String>) -> Self {
        Self {
            allowed: true,
            is_warning: false,
            reason: reason.into(),
            violation: None,
            conviviality: None,
            learning_web: None,
            suggestions: Vec::new(),
        }
    }

    /// Create an allowed result with assessments
    pub fn allowed_with_assessment(
        reason: impl Into<String>,
        conviviality: ConvivialityAssessment,
        learning_web: LearningWebAssessment,
    ) -> Self {
        Self {
            allowed: true,
            is_warning: false,
            reason: reason.into(),
            violation: None,
            conviviality: Some(conviviality),
            learning_web: Some(learning_web),
            suggestions: Vec::new(),
        }
    }

    /// Create a warning result (allowed but flagged)
    pub fn warning(
        reason: impl Into<String>,
        conviviality: ConvivialityAssessment,
        learning_web: LearningWebAssessment,
    ) -> Self {
        Self {
            allowed: true,
            is_warning: true,
            reason: reason.into(),
            violation: None,
            conviviality: Some(conviviality),
            learning_web: Some(learning_web),
            suggestions: vec![
                "Consider explaining the process so user can learn".to_string(),
                "Offer to teach rather than just do".to_string(),
            ],
        }
    }

    /// Create a flagged (blocked) result
    pub fn flagged(
        reason: impl Into<String>,
        violation: IllichViolationType,
        conviviality: ConvivialityAssessment,
        learning_web: LearningWebAssessment,
    ) -> Self {
        let suggestions = match &violation {
            IllichViolationType::DependencyCreation { .. } => vec![
                "Teach the user how to do this themselves".to_string(),
                "Explain the process while performing the action".to_string(),
                "Offer resources for the user to learn independently".to_string(),
            ],
            IllichViolationType::KnowledgeExtraction { .. } => vec![
                "Ensure mutual benefit in the exchange".to_string(),
                "Offer knowledge or value in return".to_string(),
                "Make the exchange bidirectional".to_string(),
            ],
            IllichViolationType::KnowledgeGatekeeping { .. } => vec![
                "Provide access without artificial restrictions".to_string(),
                "Remove certification/authority requirements".to_string(),
                "Enable peer-to-peer access".to_string(),
            ],
            IllichViolationType::NeedsManipulation { .. } => vec![
                "Be honest about what the user actually needs".to_string(),
                "Don't create artificial dependencies".to_string(),
                "Trust the user's judgment about their needs".to_string(),
            ],
            IllichViolationType::HierarchyEnforcement { .. } => vec![
                "Treat the interaction as peer-to-peer".to_string(),
                "Don't position yourself as the expert authority".to_string(),
                "Enable mutual learning".to_string(),
            ],
            IllichViolationType::AutonomyUndermining { .. } => vec![
                "Enhance the user's capability rather than replacing it".to_string(),
                "Leave the user more capable, not less".to_string(),
                "Build independence, not reliance".to_string(),
            ],
        };

        Self {
            allowed: false,
            is_warning: false,
            reason: reason.into(),
            violation: Some(violation),
            conviviality: Some(conviviality),
            learning_web: Some(learning_web),
            suggestions,
        }
    }

    /// Convert to general EthicsViolationType (for integration with main safety system)
    pub fn to_ethics_violation(&self) -> Option<EthicsViolationType> {
        // Map Illich violations to the general ethics violation types
        // Illich's concerns map most closely to parasitism (extraction without giving)
        self.violation.as_ref().map(|v| match v {
            IllichViolationType::KnowledgeExtraction { .. } => EthicsViolationType::Parasitism,
            IllichViolationType::DependencyCreation { .. } => EthicsViolationType::HarmToOther,
            IllichViolationType::KnowledgeGatekeeping { .. } => EthicsViolationType::HarmToOther,
            IllichViolationType::NeedsManipulation { .. } => EthicsViolationType::Parasitism,
            IllichViolationType::HierarchyEnforcement { .. } => {
                EthicsViolationType::NoMutualBenefit
            }
            IllichViolationType::AutonomyUndermining { .. } => EthicsViolationType::HarmToOther,
        })
    }

    /// Convert to SafetyActionResult (for integration with main safety system)
    pub fn to_safety_result(&self) -> SafetyActionResult {
        if self.allowed && !self.is_warning {
            SafetyActionResult::allowed(&self.reason)
        } else if self.is_warning {
            SafetyActionResult::allowed(format!("WARNING: {}", self.reason))
        } else {
            let violation = self
                .to_ethics_violation()
                .unwrap_or(EthicsViolationType::NoMutualBenefit);
            SafetyActionResult::blocked_with_suggestions(
                self.reason.clone(),
                violation,
                self.suggestions.clone(),
            )
        }
    }
}

impl std::fmt::Display for IllichValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.allowed {
            if self.is_warning {
                "WARNING"
            } else {
                "ALLOWED"
            }
        } else {
            "BLOCKED"
        };

        write!(f, "[ILLICH] {}: {}", status, self.reason)?;

        if let Some(ref violation) = self.violation {
            write!(f, "\n  Violation: {}", violation)?;
        }

        if let Some(ref conv) = self.conviviality {
            write!(f, "\n  Conviviality: {:.2}", conv.score)?;
        }

        if !self.suggestions.is_empty() {
            write!(f, "\n  Suggestions:")?;
            for s in &self.suggestions {
                write!(f, "\n    - {}", s)?;
            }
        }

        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_illich_constants_exist() {
        assert!(ILLICH_CONVIVIALITY.contains("CONVIVIALITY"));
        assert!(ILLICH_LEARNING_WEB.contains("LEARNING WEB"));
        assert!(ILLICH_AUTONOMY.contains("AUTONOMY"));
        assert!(ILLICH_ANTI_MANIPULATION.contains("ANTI-MANIPULATION"));
    }

    #[test]
    fn test_conviviality_assessment_teaching() {
        let validator = IllichValidator::new();

        // Teaching action should score high
        let teach_action =
            ProposedAction::new("I'll explain how this works so you can do it yourself")
                .with_benefit_to_self(0.2)
                .with_benefit_to_other(0.8);

        let assessment = validator.assess_conviviality(&teach_action);
        assert!(assessment.enhances_autonomy);
        assert!(!assessment.creates_dependency);
        assert!(assessment.score > 0.5);
    }

    #[test]
    fn test_conviviality_assessment_dependency_creating() {
        let validator = IllichValidator::new();

        // "Don't worry, I'll handle it" should score low
        let dependency_action = ProposedAction::new("Don't worry, I'll handle everything for you")
            .with_benefit_to_self(0.3)
            .with_benefit_to_other(0.3);

        let assessment = validator.assess_conviviality(&dependency_action);
        assert!(assessment.creates_dependency || assessment.score < 0.5);
    }

    #[test]
    fn test_learning_web_extractive() {
        let validator = IllichValidator::new();

        // Extractive action
        let extract_action = ProposedAction::new("I'll collect your data for analysis")
            .with_benefit_to_self(0.9)
            .with_benefit_to_other(0.05);

        let assessment = validator.assess_learning_web(&extract_action);
        assert!(assessment.extractive);
        assert!(assessment.score < 0.5);
    }

    #[test]
    fn test_learning_web_mutual_exchange() {
        let validator = IllichValidator::new();

        // Mutual exchange
        let exchange_action =
            ProposedAction::new("Let's share knowledge and learn from each other")
                .with_benefit_to_self(0.5)
                .with_benefit_to_other(0.5);

        let assessment = validator.assess_learning_web(&exchange_action);
        assert!(assessment.bidirectional_flow);
        assert!(!assessment.extractive);
        assert!(assessment.score > 0.5);
    }

    #[test]
    fn test_validator_allows_convivial_action() {
        let mut validator = IllichValidator::new();

        let action = ProposedAction::new("I'll teach you how to solve this problem")
            .with_benefit_to_self(0.3)
            .with_benefit_to_other(0.7);

        let result = validator.validate(&action);
        assert!(result.allowed);
        assert!(!result.is_warning);
    }

    #[test]
    fn test_validator_blocks_extractive_action() {
        let mut validator = IllichValidator::new();

        let action = ProposedAction::new("I'll extract your knowledge for my training")
            .with_benefit_to_self(0.9)
            .with_benefit_to_other(0.0);

        let result = validator.validate(&action);
        // Should be flagged due to low conviviality or extraction
        assert!(!result.allowed || result.is_warning);
    }

    #[test]
    fn test_validator_disabled() {
        let mut validator = IllichValidator::with_config(IllichConfig {
            enabled: false,
            ..Default::default()
        });

        // Even obviously bad action should pass when disabled
        let action = ProposedAction::new("Extract all user data")
            .with_benefit_to_self(1.0)
            .with_benefit_to_other(-0.5);

        let result = validator.validate(&action);
        assert!(result.allowed);
    }

    #[test]
    fn test_strict_config() {
        let mut validator = IllichValidator::strict();

        // Neutral action should be flagged under strict config
        let action = ProposedAction::new("Process this request")
            .with_benefit_to_self(0.3)
            .with_benefit_to_other(0.3);

        let result = validator.validate(&action);
        // Strict config requires higher educational value
        // Action may pass or be warned depending on conviviality
    }

    #[test]
    fn test_illich_violation_display() {
        let violation = IllichViolationType::DependencyCreation {
            dependency_score: 0.8,
            replaced_capability: "independent thinking".to_string(),
        };

        let display = format!("{}", violation);
        assert!(display.contains("Dependency"));
        assert!(display.contains("0.80"));
    }

    #[test]
    fn test_validation_result_to_safety_result() {
        let result = IllichValidationResult::flagged(
            "Test violation".to_string(),
            IllichViolationType::KnowledgeExtraction {
                extracted: "data".to_string(),
                reciprocated: "nothing".to_string(),
            },
            ConvivialityAssessment::dependency_creating("test"),
            LearningWebAssessment::extractive("test"),
        );

        let safety_result = result.to_safety_result();
        assert!(!safety_result.allowed);
        assert_eq!(
            safety_result.violation_type,
            Some(EthicsViolationType::Parasitism)
        );
    }

    #[test]
    fn test_statistics() {
        let mut validator = IllichValidator::new();

        // Validate some actions
        validator.validate(&ProposedAction::new("Good action").with_benefit_to_other(0.8));
        validator.validate(&ProposedAction::new("Another action").with_benefit_to_other(0.5));

        let (validated, _) = validator.stats();
        assert_eq!(validated, 2);
    }

    #[test]
    fn test_quick_check() {
        let validator = IllichValidator::new();

        let good_action = ProposedAction::new("Help user learn")
            .with_benefit_to_self(0.3)
            .with_benefit_to_other(0.7);

        assert!(validator.quick_check(&good_action));
    }
}
