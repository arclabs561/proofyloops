use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Planner output: a strictly-typed “attention + budget” decision for one expansion step.
///
/// This is intentionally small and falsifiable: same evidence -> same JSON -> same effect.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
pub struct PlannerDecision {
    /// Confidence in the decision (0..=1). If low, caller should fall back to heuristics.
    pub confidence: f64,

    /// Optional: recommend a specific hole (line number, 1-indexed) to focus next.
    /// If null, caller keeps its current selection heuristic.
    #[schemars(required)]
    #[serde(default)]
    pub focus_line_1: Option<u64>,

    /// Optional: override oracle passes (1..=4). If null, caller keeps defaults.
    #[schemars(required)]
    #[serde(default)]
    pub oracle_passes: Option<u64>,

    /// Optional: a preferred oracle tactic ordering (subset of ["simp?","exact?","apply?","aesop?"]).
    /// If null/empty, caller uses defaults.
    #[schemars(required)]
    #[serde(default)]
    pub oracle_tactics: Vec<String>,

    /// Optional: recommend skipping some oracle tactics (e.g. ["aesop?"]).
    #[schemars(required)]
    #[serde(default)]
    pub ban_oracle_tactics: Vec<String>,

    /// Short rationale for humans (bounded).
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
pub struct PlannerEvidence {
    /// Stable goal-state key if available; 0 may mean unknown.
    pub state_key: u64,
    /// Goal summary (single goal pretty line, and counts).
    pub goal: PlannerGoalSummary,
    /// Candidate holes to choose from (bounded, line numbers).
    pub candidate_holes: Vec<PlannerHole>,
    /// Current budgets/knobs.
    pub config: PlannerConfigSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
pub struct PlannerGoalSummary {
    pub target: String,
    pub n_goals: u64,
    pub hyps_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
pub struct PlannerHole {
    pub line_1: u64,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
pub struct PlannerConfigSummary {
    pub goal_first_k: u64,
    pub oracle_max_calls: u64,
    pub timeout_s: u64,
}

/// Run the planner as a one-shot LLM call returning strict JSON.
///
/// NOTE: this intentionally uses the same provider routing as `llm_summary` and does not run tools.
pub async fn plan(
    system: &str,
    evidence_json: &str,
    timeout: std::time::Duration,
) -> Result<(PlannerDecision, String), String> {
    let done = crate::llm::chat_completion(system, evidence_json, timeout).await?;
    let parsed: PlannerDecision = serde_json::from_str(&done.content)
        .map_err(|e| format!("planner json parse failed: {e}"))?;
    Ok((parsed, done.content))
}

