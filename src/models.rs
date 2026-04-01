//! Core data models for the Hindsight memory architecture.
//!
//! Defines the four memory networks, graph edges, the agent profile, and
//! all intermediate structs used during fact extraction and retrieval.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The four logical memory networks.
///
/// Each network stores a different category of knowledge:
///
/// | Variant | Stores |
/// | ---------|------- |
/// | `World` | Objective facts about the external world |
/// | `Experience` | Biographical information about the agent (first-person) |
/// | `Opinion` | Subjective judgments with a confidence score |
/// | `Observation` | Preference-neutral synthesized summaries of entities |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetworkType {
    /// Objective facts about the external world.
    World,
    /// Biographical information about the agent (first-person).
    Experience,
    /// Subjective judgments with an associated confidence score.
    Opinion,
    /// Preference-neutral synthesized summaries of entities.
    Observation,
}

impl NetworkType {
    /// Returns the string tag stored in the database.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::World => "world",
            Self::Experience => "experience",
            Self::Opinion => "opinion",
            Self::Observation => "observation",
        }
    }

    /// Parses a network tag string. Returns `None` for unknown values.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "world" => Some(Self::World),
            "experience" => Some(Self::Experience),
            "opinion" => Some(Self::Opinion),
            "observation" => Some(Self::Observation),
            _ => None,
        }
    }
}

/// Types of relationships between memory units in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeType {
    /// Sequential or time-based relationship.
    Temporal,
    /// Meaning-based similarity relationship.
    Semantic,
    /// Shared-entity reference relationship.
    Entity,
    /// Cause-and-effect relationship.
    Causal,
}

impl EdgeType {
    /// Returns the string tag stored in the database.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Temporal => "temporal",
            Self::Semantic => "semantic",
            Self::Entity => "entity",
            Self::Causal => "causal",
        }
    }

    /// Parses an edge-type tag string. Returns `None` for unknown values.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "temporal" => Some(Self::Temporal),
            "semantic" => Some(Self::Semantic),
            "entity" => Some(Self::Entity),
            "causal" => Some(Self::Causal),
            _ => None,
        }
    }
}

/// A single unit of memory stored in one of the four networks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUnit {
    /// Unique identifier.
    pub id: Uuid,
    /// Which network this memory belongs to.
    pub network: NetworkType,
    /// The narrative content of the memory.
    pub content: String,
    /// Pre-computed embedding vector for semantic search.
    pub embedding: Vec<f32>,
    /// Named entities extracted from the content.
    pub entities: Vec<String>,
    /// Confidence score (only set for [`NetworkType::Opinion`] memories).
    pub confidence: Option<f32>,
    /// When the memory was first created.
    pub created_at: DateTime<Utc>,
    /// When the memory was last updated.
    pub updated_at: DateTime<Utc>,
}

/// A directed edge between two memory units in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier.
    pub id: Uuid,
    /// Source memory unit.
    pub source_id: Uuid,
    /// Target memory unit.
    pub target_id: Uuid,
    /// Kind of relationship.
    pub edge_type: EdgeType,
    /// Strength of the relationship (0.0–1.0).
    pub weight: f32,
    /// When the edge was created.
    pub created_at: DateTime<Utc>,
}

/// Behavioral profile that shapes the agent's responses during reflection.
///
/// Disposition parameters range from 1 to 5 (except `bias_strength` which is
/// 0.0–1.0) and are injected into the LLM system prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Display name of the agent.
    pub name: String,
    /// Background description.
    pub background: String,
    /// How much the agent questions claims (1 = trusting, 5 = highly skeptical).
    pub skepticism: u8,
    /// How literally the agent interprets statements (1 = figurative, 5 = strictly literal).
    pub literalism: u8,
    /// How much the agent considers others' feelings (1 = detached, 5 = highly empathetic).
    pub empathy: u8,
    /// How strongly existing opinions influence new responses (0.0–1.0).
    pub bias_strength: f32,
}

/// A single fact extracted from a conversation by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// Self-contained narrative text of the fact.
    pub content: String,
    /// Which network this fact should be stored in.
    pub network: NetworkType,
    /// Named entities mentioned in the fact.
    pub entities: Vec<String>,
    /// Confidence score (present only for [`NetworkType::Opinion`] facts).
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Links to other facts in the same extraction batch.
    #[serde(default)]
    pub links: Vec<FactLink>,
}

/// A relationship between two facts extracted in the same batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactLink {
    /// Zero-based index of the target fact within the batch.
    pub target_fact_index: usize,
    /// Kind of relationship.
    pub edge_type: EdgeType,
}

/// Wrapper returned by the LLM fact-extraction prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFacts {
    /// The list of extracted facts.
    pub facts: Vec<ExtractedFact>,
}

/// A memory unit paired with a relevance score from retrieval.
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    /// The memory unit.
    pub memory: MemoryUnit,
    /// Relevance or fusion score (higher is more relevant).
    pub score: f64,
}
