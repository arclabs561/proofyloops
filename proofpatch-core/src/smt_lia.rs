//! Minimal SMT-based entailment checks for linear integer arithmetic (LIA).
//!
//! This is intentionally conservative and best-effort:
//! - If we cannot confidently parse/sort variables, return `Ok(None)`.
//! - Uses an external SMT solver via `smtkit` if available; if none available, return `Ok(None)`.
//!
//! Soundness posture: this is a *heuristic signal* for ranking / candidate selection.
//! It must never be used as a proof of a Lean goal without verification.

use regex::Regex;
use serde_json::Value;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

static SMT_SOLVER_PROBE: OnceLock<Value> = OnceLock::new();

/// Probe whether `smtkit` can spawn a solver session (cached per-process).
///
/// This is intended for observability and install debugging: if no solver is on `PATH`,
/// `smtkit` will report which command lines it tried.
pub fn smt_solver_probe() -> Value {
    SMT_SOLVER_PROBE
        .get_or_init(|| match smtkit::session::spawn_auto_with_caps() {
            Ok((sess, used, caps)) => {
                // Best-effort terminate gracefully (drop would kill anyway).
                let _ = sess.exit();
                serde_json::json!({
                    "available": true,
                    "used": used,
                    "caps": {
                        "check_sat_assuming": caps.check_sat_assuming,
                        "get_model": caps.get_model,
                        "get_unsat_core": caps.get_unsat_core,
                        "get_proof": caps.get_proof,
                        "named_assertions_in_core": caps.named_assertions_in_core,
                    }
                })
            }
            Err(e) => serde_json::json!({
                "available": false,
                "error": e.to_string(),
            }),
        })
        .clone()
}

/// A reusable “warm” SMT session for repeated entailment checks.
///
/// This is intentionally **best-effort**:
/// - If no solver is available, callers should fall back to non-SMT heuristics.
/// - We only use `check-sat-assuming` to avoid accumulating assertions.
pub struct ReusableSmtSession {
    sess: smtkit::session::SmtlibSession,
    solver_used: String,
    // vars we already declared in the session
    declared: std::collections::BTreeSet<String>,
    // Nat vars for which we should include `x >= 0` in assumptions
    nat_vars: std::collections::BTreeSet<String>,
    // Whether the underlying solver supports check-sat-assuming.
    supports_assuming: bool,
    // Simple counters for observability.
    decls_added: u64,
    checks_assuming: u64,
    checks_pushpop: u64,
    fragment_hits: u64,
    fragment_misses: u64,
    fragment_resets: u64,
    fragment_active_key: Option<u64>,
    max_fragment_assert_terms: usize,
    max_assumptions_terms: usize,
    unknowns: u64,
    last_reason_unknown: Option<String>,
    disabled: bool,
    disabled_reason: Option<String>,
    errors: u64,
}

fn reuse_fragment_key(
    nat_in_use: &std::collections::BTreeSet<String>,
    hyps: &[ParsedRelConstraint],
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // Nat vars included (sorted due to BTreeSet).
    for v in nat_in_use.iter() {
        v.hash(&mut h);
        0xffu8.hash(&mut h);
    }
    0xfeu8.hash(&mut h);
    // Hypotheses included (order matters; we keep it stable).
    for hyp in hyps.iter() {
        hyp.src.hash(&mut h);
        0xfdu8.hash(&mut h);
    }
    h.finish()
}

impl ReusableSmtSession {
    pub fn new() -> Result<Option<Self>, String> {
        let (mut sess, used, caps) = match smtkit::session::spawn_auto_with_caps() {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        sess.set_logic("QF_LIA").map_err(|e| e.to_string())?;
        sess.set_print_success(false).map_err(|e| e.to_string())?;
        sess.set_produce_models(false).map_err(|e| e.to_string())?;

        let supports_assuming = caps.check_sat_assuming;
        // Reuse requires `check-sat-assuming` (we use it to avoid accumulating assertions).
        // Historically we disabled reuse for some solvers by name due to session brittleness,
        // but we now prefer a capability-first posture and allow reuse to self-disable on errors.
        let reuse_disabled_for_solver = !supports_assuming;

        Ok(Some(Self {
            sess,
            solver_used: used,
            declared: std::collections::BTreeSet::new(),
            nat_vars: std::collections::BTreeSet::new(),
            supports_assuming,
            decls_added: 0,
            checks_assuming: 0,
            checks_pushpop: 0,
            fragment_hits: 0,
            fragment_misses: 0,
            fragment_resets: 0,
            fragment_active_key: None,
            max_fragment_assert_terms: 0,
            max_assumptions_terms: 0,
            unknowns: 0,
            last_reason_unknown: None,
            disabled: reuse_disabled_for_solver,
            disabled_reason: if reuse_disabled_for_solver {
                Some("reuse_disabled_for_solver_missing_check_sat_assuming".to_string())
            } else {
                None
            },
            errors: 0,
        }))
    }

    pub fn stats(&self) -> Value {
        serde_json::json!({
            "solver": self.solver_used,
            "supports_check_sat_assuming": self.supports_assuming,
            "declared_vars": self.declared.len(),
            "nat_vars": self.nat_vars.len(),
            "decls_added": self.decls_added,
            "checks_assuming": self.checks_assuming,
            "checks_pushpop": self.checks_pushpop,
            "fragment_hits": self.fragment_hits,
            "fragment_misses": self.fragment_misses,
            "fragment_resets": self.fragment_resets,
            "max_fragment_assert_terms": self.max_fragment_assert_terms,
            "max_assumptions_terms": self.max_assumptions_terms,
            "unknowns": self.unknowns,
            "last_reason_unknown": self.last_reason_unknown,
            "disabled": self.disabled,
            "disabled_reason": self.disabled_reason,
            "errors": self.errors,
        })
    }

    fn disable(&mut self, reason: String) {
        self.errors = self.errors.saturating_add(1);
        self.disabled = true;
        self.disabled_reason = Some(reason);
        // Best-effort: kill the underlying process so we don't keep a wedged solver around.
        self.sess.kill();
    }

    fn ensure_declared(
        &mut self,
        used_vars: &std::collections::BTreeSet<String>,
        var_kinds: &std::collections::BTreeMap<String, VarKind>,
    ) -> Result<(), String> {
        for name in used_vars.iter() {
            if self.declared.contains(name) {
                continue;
            }
            self.sess
                .declare_const(name, &smtkit::smt2::Sort::Int.to_smt2())
                .map_err(|e| e.to_string())?;
            self.declared.insert(name.clone());
            self.decls_added = self.decls_added.saturating_add(1);
        }
        // Track Nat vars so we can enforce nonnegativity via assumptions.
        for (name, kind) in var_kinds.iter() {
            if *kind == VarKind::Nat && used_vars.contains(name) {
                self.nat_vars.insert(name.clone());
            }
        }
        Ok(())
    }

    fn ensure_fragment_asserted(
        &mut self,
        hyps: &[ParsedRelConstraint],
        used_vars: &std::collections::BTreeSet<String>,
    ) -> Result<(), String> {
        use smtkit::smt2::t;

        // Which Nat vars should be constrained for this fragment?
        let mut nat_in_use: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for v in self.nat_vars.iter() {
            if used_vars.contains(v) {
                nat_in_use.insert(v.clone());
            }
        }

        let key = reuse_fragment_key(&nat_in_use, hyps);
        if self.fragment_active_key == Some(key) {
            self.fragment_hits = self.fragment_hits.saturating_add(1);
            return Ok(());
        }
        self.fragment_misses = self.fragment_misses.saturating_add(1);

        // Reset the active fragment frame (if any).
        if self.fragment_active_key.is_some() {
            self.sess.pop(1).map_err(|e| e.to_string())?;
            self.fragment_resets = self.fragment_resets.saturating_add(1);
            self.fragment_active_key = None;
        }

        // Install a new fragment frame with Nat constraints + hypotheses asserted.
        self.sess.push().map_err(|e| e.to_string())?;

        let mut asserted_terms: usize = 0;
        for v in nat_in_use.iter() {
            self.sess
                .assert_sexp(&t::ge(t::sym(v.clone()), t::int_lit(0)))
                .map_err(|e| e.to_string())?;
            asserted_terms = asserted_terms.saturating_add(1);
        }
        for h in hyps.iter() {
            self.sess.assert_sexp(&h.sexp).map_err(|e| e.to_string())?;
            asserted_terms = asserted_terms.saturating_add(1);
        }

        self.max_fragment_assert_terms = self.max_fragment_assert_terms.max(asserted_terms);
        self.fragment_active_key = Some(key);
        Ok(())
    }

    fn check_entails_pushpop(
        &mut self,
        timeout_ms: u64,
        seed: u64,
        hyps: &[ParsedRelConstraint],
        target: &ParsedRelConstraint,
        used_vars: &std::collections::BTreeSet<String>,
    ) -> Result<Option<bool>, String> {
        use smtkit::smt2::t;
        if self.disabled {
            return Ok(None);
        }

        self.sess
            .set_timeout_ms(timeout_ms)
            .map_err(|e| e.to_string())?;
        self.sess.set_random_seed(seed).map_err(|e| e.to_string())?;

        self.ensure_fragment_asserted(hyps, used_vars)?;

        // Push an inner frame for just `¬target`, then pop it.
        self.sess.push().map_err(|e| e.to_string())?;
        self.sess
            .assert_sexp(&t::not(target.sexp.clone()))
            .map_err(|e| e.to_string())?;
        self.checks_pushpop = self.checks_pushpop.saturating_add(1);
        let st = self.sess.check_sat().map_err(|e| e.to_string())?;
        self.sess.pop(1).map_err(|e| e.to_string())?;

        Ok(match st {
            smtkit::session::Status::Unsat => Some(true),
            smtkit::session::Status::Sat => Some(false),
            smtkit::session::Status::Unknown => {
                self.unknowns = self.unknowns.saturating_add(1);
                self.last_reason_unknown = self
                    .sess
                    .get_info(":reason-unknown")
                    .ok()
                    .map(|s| s.to_string());
                None
            }
        })
    }

    fn check_entails_assuming(
        &mut self,
        timeout_ms: u64,
        seed: u64,
        hyps: &[ParsedRelConstraint],
        target: &ParsedRelConstraint,
        used_vars: &std::collections::BTreeSet<String>,
    ) -> Result<Option<bool>, String> {
        use smtkit::smt2::t;
        if self.disabled {
            return Ok(None);
        }
        if !self.supports_assuming {
            return Ok(None);
        }
        self.sess
            .set_timeout_ms(timeout_ms)
            .map_err(|e| e.to_string())?;
        self.sess.set_random_seed(seed).map_err(|e| e.to_string())?;

        // Assumptions: Nat constraints + all hyp constraints + ¬target.
        let mut assumptions: Vec<smtkit::sexp::Sexp> = Vec::new();
        for v in self.nat_vars.iter() {
            if used_vars.contains(v) {
                assumptions.push(t::ge(t::sym(v.clone()), t::int_lit(0)));
            }
        }
        for h in hyps {
            assumptions.push(h.sexp.clone());
        }
        assumptions.push(t::not(target.sexp.clone()));
        self.checks_assuming = self.checks_assuming.saturating_add(1);
        self.max_assumptions_terms = self.max_assumptions_terms.max(assumptions.len());

        let st = self
            .sess
            .check_sat_assuming(&assumptions)
            .map_err(|e| e.to_string())?;
        Ok(match st {
            smtkit::session::Status::Unsat => Some(true),
            smtkit::session::Status::Sat => Some(false),
            smtkit::session::Status::Unknown => {
                self.unknowns = self.unknowns.saturating_add(1);
                self.last_reason_unknown = self
                    .sess
                    .get_info(":reason-unknown")
                    .ok()
                    .map(|s| s.to_string());
                None
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarKind {
    Int,
    Nat,
}

fn sanitize_name(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out = "x".to_string();
    }
    if out
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        out.insert(0, '_');
    }
    out
}

fn extract_decl_kind(hyp_text: &str) -> Option<(String, VarKind)> {
    // Recognize tiny declaration shapes like:
    // - `n : ℕ` / `n : Nat`
    // - `m : ℤ` / `m : Int`
    let (name, ty) = hyp_text.split_once(':')?;
    let name = name.trim();
    let ty = ty.trim();
    if name.is_empty() || ty.is_empty() {
        return None;
    }
    let kind = if ty.contains('ℕ') || ty.contains("Nat") {
        VarKind::Nat
    } else if ty.contains('ℤ') || ty.contains("Int") {
        VarKind::Int
    } else {
        return None;
    };
    Some((sanitize_name(name), kind))
}

#[derive(Debug, Clone)]
struct LinearExpr {
    // var -> coefficient
    coeffs: std::collections::BTreeMap<String, i64>,
    c0: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelOp {
    Le,
    Ge,
    Lt,
    Gt,
    Eq,
}

#[derive(Debug, Clone)]
struct ParsedRel {
    op: RelOp,
    lhs: LinearExpr,
    rhs: LinearExpr,
}

fn parse_linear_expr_int(s: &str) -> Option<LinearExpr> {
    // Small parser: sums/differences of identifiers and integer literals.
    // Rejects obvious non-LIA operators.
    let bad = ['*', '/', '^', '·', '↑', '∑', '∏'];
    if s.chars().any(|c| bad.contains(&c)) {
        return None;
    }
    let mut coeffs: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
    let mut c0: i64 = 0;
    let mut i = 0usize;
    let chars: Vec<char> = s.chars().collect();
    let mut sign: i64 = 1;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '+' {
            sign = 1;
            i += 1;
            continue;
        }
        if ch == '-' {
            sign = -1;
            i += 1;
            continue;
        }
        if ch.is_ascii_digit() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let lit: String = chars[i..j].iter().collect();
            let v: i64 = lit.parse().ok()?;
            c0 = c0.saturating_add(sign.saturating_mul(v));
            i = j;
            continue;
        }
        if ch.is_alphanumeric() || ch == '_' || ch == '.' {
            let mut j = i + 1;
            while j < chars.len()
                && (chars[j].is_alphanumeric() || chars[j] == '_' || chars[j] == '.')
            {
                j += 1;
            }
            let raw: String = chars[i..j].iter().collect();
            let name = sanitize_name(&raw);
            *coeffs.entry(name.clone()).or_insert(0) =
                coeffs.get(&name).copied().unwrap_or(0).saturating_add(sign);
            i = j;
            continue;
        }
        return None;
    }
    Some(LinearExpr { coeffs, c0 })
}

#[derive(Debug, Clone)]
struct ParsedRelConstraint {
    rel: ParsedRel,
    sexp: smtkit::sexp::Sexp,
    vars: std::collections::BTreeSet<String>,
    src: String,
}

fn select_constraints_by_var_depth(
    target_vars: &std::collections::BTreeSet<String>,
    hyps: &[ParsedRelConstraint],
    depth: usize,
) -> Vec<ParsedRelConstraint> {
    // Depth semantics (deterministic, bounded):
    // - depth == 0: return all hyps (status quo behavior)
    // - depth == 1: include only hyps mentioning a target var
    // - depth == 2+: expand by variable connectivity up to `depth` passes
    //
    // This is a performance/robustness knob, not a soundness boundary: SMT here is only a heuristic signal.
    if depth == 0 {
        return hyps.to_vec();
    }
    if target_vars.is_empty() || hyps.is_empty() {
        return Vec::new();
    }

    let mut included_vars: std::collections::BTreeSet<String> = target_vars.clone();
    let mut picked: Vec<bool> = vec![false; hyps.len()];

    // One pass per "hop" from the target vars.
    for _ in 0..depth {
        // Snapshot the current frontier so we don't chain within one hop.
        // (We only expand `included_vars` *after* finishing the pass.)
        let frontier = included_vars.clone();
        let mut pending_vars: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        let mut changed = false;
        for (i, h) in hyps.iter().enumerate() {
            if picked[i] {
                continue;
            }
            let touches = h.vars.iter().any(|v| frontier.contains(v));
            if touches {
                picked[i] = true;
                pending_vars.extend(h.vars.iter().cloned());
                changed = true;
            }
        }
        included_vars.extend(pending_vars);
        if !changed {
            break;
        }
    }

    let mut out: Vec<ParsedRelConstraint> = Vec::new();
    for (i, h) in hyps.iter().enumerate() {
        if picked[i] {
            out.push(h.clone());
        }
    }
    out
}

fn linear_expr_to_smt_sexp(e: &LinearExpr) -> smtkit::sexp::Sexp {
    use smtkit::smt2::t;
    let mut terms: Vec<smtkit::sexp::Sexp> = Vec::new();
    if e.c0 != 0 {
        terms.push(t::int_lit(e.c0));
    }
    for (v, c) in e.coeffs.iter() {
        if *c == 0 {
            continue;
        }
        let sym = t::sym(v.clone());
        if *c == 1 {
            terms.push(sym);
        } else if *c == -1 {
            terms.push(t::app("-", vec![sym]));
        } else {
            terms.push(t::app("*", vec![t::int_lit(*c), sym]));
        }
    }
    if terms.is_empty() {
        t::int_lit(0)
    } else if terms.len() == 1 {
        terms[0].clone()
    } else {
        t::add(terms)
    }
}

fn normalize_expr_text(s: &str) -> String {
    // Try to normalize a few common pretty-printed shapes into linear syntax.
    // This is intentionally small; if it doesn't match, we fall back to "unknown".
    let mut out = s.to_string();
    // Nat.succ x  -> x + 1
    if let Ok(re1) = Regex::new(r"\bNat\.succ\s+\(?([A-Za-z0-9_\.]+)\)?") {
        out = re1.replace_all(&out, "$1 + 1").to_string();
    }
    // Int.succ x -> x + 1
    if let Ok(re2) = Regex::new(r"\bInt\.succ\s+\(?([A-Za-z0-9_\.]+)\)?") {
        out = re2.replace_all(&out, "$1 + 1").to_string();
    }
    // Drop parentheses which often wrap pretty-printed terms.
    out = out
        .chars()
        .map(|c| if c == '(' || c == ')' { ' ' } else { c })
        .collect();
    out
}

fn parse_rel_constraint_int(s: &str) -> Option<ParsedRelConstraint> {
    let s = s.trim();
    let src = s.to_string();
    let ops = ["<=", "≤", ">=", "≥", "<", ">", "="];
    let (op, idx) = ops.iter().find_map(|op| s.find(op).map(|i| (*op, i)))?;
    let (lhs, rhs0) = s.split_at(idx);
    let rhs = rhs0.get(op.len()..)?;
    let lhs_n = normalize_expr_text(lhs.trim());
    let rhs_n = normalize_expr_text(rhs.trim());
    let lhs_e = parse_linear_expr_int(lhs_n.trim())?;
    let rhs_e = parse_linear_expr_int(rhs_n.trim())?;
    let rel_op = match op {
        "<=" | "≤" => RelOp::Le,
        ">=" | "≥" => RelOp::Ge,
        "<" => RelOp::Lt,
        ">" => RelOp::Gt,
        "=" => RelOp::Eq,
        _ => return None,
    };
    use smtkit::smt2::t;
    let a = linear_expr_to_smt_sexp(&lhs_e);
    let b = linear_expr_to_smt_sexp(&rhs_e);
    let sexp = match rel_op {
        RelOp::Le => t::le(a, b),
        RelOp::Ge => t::ge(a, b),
        RelOp::Lt => t::lt(a, b),
        RelOp::Gt => t::app(">", vec![a, b]),
        RelOp::Eq => t::eq(a, b),
    };
    let mut vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    vars.extend(lhs_e.coeffs.keys().cloned());
    vars.extend(rhs_e.coeffs.keys().cloned());
    Some(ParsedRelConstraint {
        rel: ParsedRel {
            op: rel_op,
            lhs: lhs_e,
            rhs: rhs_e,
        },
        sexp,
        vars,
        src,
    })
}

fn entails_by_offset_addition(target: &ParsedRel, hyps: &[ParsedRelConstraint]) -> Option<bool> {
    if target.op != RelOp::Le {
        return None;
    }
    for h in hyps {
        if h.rel.op != RelOp::Le {
            continue;
        }
        if target.lhs.coeffs != h.rel.lhs.coeffs {
            continue;
        }
        if target.rhs.coeffs != h.rel.rhs.coeffs {
            continue;
        }
        let dl = target.lhs.c0.saturating_sub(h.rel.lhs.c0);
        let dr = target.rhs.c0.saturating_sub(h.rel.rhs.c0);
        if dl == dr {
            return Some(true);
        }
    }
    None
}

fn entails_by_drop_nonneg_var(target: &ParsedRel, hyps: &[ParsedRelConstraint]) -> Option<bool> {
    // Tiny deterministic LIA heuristic:
    //
    // If we have:
    // - hyp1: lhs + v <= rhs
    // - hyp2: 0 <= v   (or v >= 0)
    //
    // then infer: lhs <= rhs.
    //
    // This is strictly a best-effort entailment *proof* heuristic. It never proves non-entailment.
    if target.op != RelOp::Le {
        return None;
    }

    // Collect vars known to be >= 0 from hyps.
    let mut nonneg: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    fn is_zero_expr(e: &LinearExpr) -> bool {
        e.c0 == 0 && e.coeffs.is_empty()
    }

    for h in hyps.iter() {
        match h.rel.op {
            RelOp::Ge | RelOp::Le => { /* handled below */ }
            _ => continue,
        }

        // Normalize to `expr >= 0` form by looking for "v >= 0" or "0 <= v".
        // We only accept the very simple case: a single var with coefficient 1 and c0 = 0.
        let (lhs, rhs) = (&h.rel.lhs, &h.rel.rhs);
        let var_expr = if h.rel.op == RelOp::Ge && is_zero_expr(rhs) {
            Some(lhs)
        } else if h.rel.op == RelOp::Le && is_zero_expr(lhs) {
            Some(rhs)
        } else {
            None
        };

        if let Some(e) = var_expr {
            if e.c0 == 0 && e.coeffs.len() == 1 {
                if let Some((v, c)) = e.coeffs.iter().next() {
                    if *c == 1 {
                        nonneg.insert(v.clone());
                    }
                }
            }
        }
    }
    if nonneg.is_empty() {
        return None;
    }

    // For each hyp, check whether it is target + v where v is known nonnegative.
    let target_e = linear_sub(&target.lhs, &target.rhs); // target_e <= 0
    for h in hyps.iter() {
        if h.rel.op != RelOp::Le {
            continue;
        }
        let hyp_e = linear_sub(&h.rel.lhs, &h.rel.rhs); // hyp_e <= 0
                                                        // Want: hyp_e = target_e + v  =>  (hyp_e - target_e) = v
        let t = linear_sub(&hyp_e, &target_e);
        if t.c0 != 0 || t.coeffs.len() != 1 {
            continue;
        }
        if let Some((v, c)) = t.coeffs.iter().next() {
            if *c == 1 && nonneg.contains(v) {
                return Some(true);
            }
        }
    }

    None
}

#[derive(Debug, Clone)]
struct IdlEdge {
    // Represents constraint: to <= from + w
    from: String,
    to: String,
    w: i64,
}

fn linear_sub(a: &LinearExpr, b: &LinearExpr) -> LinearExpr {
    let mut coeffs = a.coeffs.clone();
    for (v, c) in b.coeffs.iter() {
        let cur = coeffs.get(v).copied().unwrap_or(0);
        let next = cur.saturating_sub(*c);
        if next == 0 {
            coeffs.remove(v);
        } else {
            coeffs.insert(v.clone(), next);
        }
    }
    LinearExpr {
        coeffs,
        c0: a.c0.saturating_sub(b.c0),
    }
}

fn idl_edges_from_rel(r: &ParsedRel) -> Option<Vec<IdlEdge>> {
    // Extract difference-logic constraints from a relation when possible.
    //
    // We only use these edges to *prove entailment* (never to prove non-entailment),
    // so it is safe to ignore constraints we cannot encode.
    match r.op {
        RelOp::Le => {
            // lhs - rhs <= 0  ==>  sum_i coeff_i*v_i + c0 <= 0
            let e = linear_sub(&r.lhs, &r.rhs);
            idl_edges_from_linear_leq0(&e)
        }
        RelOp::Ge => {
            // lhs >= rhs  <=>  rhs - lhs <= 0
            let e = linear_sub(&r.rhs, &r.lhs);
            idl_edges_from_linear_leq0(&e)
        }
        RelOp::Eq => {
            // lhs = rhs  <=>  lhs <= rhs AND rhs <= lhs
            let mut out = Vec::new();
            let e1 = linear_sub(&r.lhs, &r.rhs);
            let e2 = linear_sub(&r.rhs, &r.lhs);
            if let Some(xs) = idl_edges_from_linear_leq0(&e1) {
                out.extend(xs);
            } else {
                return None;
            }
            if let Some(xs) = idl_edges_from_linear_leq0(&e2) {
                out.extend(xs);
            } else {
                return None;
            }
            Some(out)
        }
        RelOp::Lt | RelOp::Gt => None,
    }
}

fn idl_edges_from_linear_leq0(e: &LinearExpr) -> Option<Vec<IdlEdge>> {
    // e = sum coeff_i*v_i + c0 <= 0
    // Recognize shapes:
    // - x - y + k <= 0  => x - y <= -k  => edge y -> x (w = -k)
    // - x + k <= 0      => x <= -k      => edge 0 -> x (w = -k)
    // - -x + k <= 0     => x >= k       => 0 - x <= -k => edge x -> 0 (w = -k)
    let nz: Vec<(String, i64)> = e
        .coeffs
        .iter()
        .filter_map(|(v, c)| if *c != 0 { Some((v.clone(), *c)) } else { None })
        .collect();
    if nz.is_empty() {
        // pure constant constraint: c0 <= 0; if violated, hyps are inconsistent.
        return Some(Vec::new());
    }
    if nz.len() == 1 {
        let (v, c) = (&nz[0].0, nz[0].1);
        if c == 1 {
            // x + c0 <= 0 -> x <= -c0
            return Some(vec![IdlEdge {
                from: "__ZERO__".to_string(),
                to: v.clone(),
                w: -e.c0,
            }]);
        }
        if c == -1 {
            // -x + c0 <= 0 -> x >= c0  -> 0 - x <= -c0  -> x -> 0, w=-c0
            return Some(vec![IdlEdge {
                from: v.clone(),
                to: "__ZERO__".to_string(),
                w: -e.c0,
            }]);
        }
        return None;
    }
    if nz.len() == 2 {
        let (v1, c1) = (&nz[0].0, nz[0].1);
        let (v2, c2) = (&nz[1].0, nz[1].1);
        if c1 == 1 && c2 == -1 {
            // v1 - v2 + c0 <= 0 -> v1 - v2 <= -c0 -> edge v2 -> v1, w=-c0
            return Some(vec![IdlEdge {
                from: v2.clone(),
                to: v1.clone(),
                w: -e.c0,
            }]);
        }
        if c1 == -1 && c2 == 1 {
            // v2 - v1 + c0 <= 0 -> v2 - v1 <= -c0 -> edge v1 -> v2, w=-c0
            return Some(vec![IdlEdge {
                from: v1.clone(),
                to: v2.clone(),
                w: -e.c0,
            }]);
        }
    }
    None
}

fn idl_proves_entails(
    target: &ParsedRelConstraint,
    hyps: &[ParsedRelConstraint],
    var_kinds: &std::collections::BTreeMap<String, VarKind>,
) -> Option<bool> {
    // Only ever returns Some(true) (proved) or None (unknown).
    // We do not attempt to prove non-entailment with IDL because the encoding is partial.
    let target_edges = idl_edges_from_rel(&target.rel)?;
    if target_edges.is_empty() {
        // target is a pure constant relation; treat it as unknown.
        return None;
    }
    // Build node set.
    let mut nodes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    nodes.insert("__ZERO__".to_string());
    for (name, _) in var_kinds.iter() {
        nodes.insert(name.clone());
    }
    // We also need vars appearing in parsed hyps (which are subset of var_kinds by construction).
    for h in hyps {
        for v in h.vars.iter() {
            nodes.insert(v.clone());
        }
    }
    for v in target.vars.iter() {
        nodes.insert(v.clone());
    }

    let mut idx: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (i, n) in nodes.iter().enumerate() {
        idx.insert(n.clone(), i);
    }
    let n = idx.len();
    if n == 0 {
        return None;
    }

    let mut edges: Vec<(usize, usize, i64)> = Vec::new();
    // Nat nonneg constraints: x >= 0  -> edge x -> 0 with w=0 in our representation.
    for (name, kind) in var_kinds.iter() {
        if *kind == VarKind::Nat {
            if let (Some(&u), Some(&v0)) = (idx.get(name), idx.get("__ZERO__")) {
                edges.push((u, v0, 0));
            }
        }
    }
    // Hypothesis edges (best-effort; skip unencodable ones).
    for h in hyps {
        if let Some(es) = idl_edges_from_rel(&h.rel) {
            for e in es {
                let from = *idx.get(&e.from)?;
                let to = *idx.get(&e.to)?;
                edges.push((from, to, e.w));
            }
        }
    }

    // If constraints are inconsistent (negative cycle), they entail everything.
    // Detect via Bellman-Ford from a super-source connected to all nodes with 0 edges.
    let mut dist = vec![0i64; n];
    for _ in 0..n {
        let mut changed = false;
        for (u, v, w) in edges.iter().copied() {
            let cand = dist[u].saturating_add(w);
            if cand < dist[v] {
                dist[v] = cand;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // One more pass: if any relaxes, negative cycle exists.
    for (u, v, w) in edges.iter().copied() {
        let cand = dist[u].saturating_add(w);
        if cand < dist[v] {
            return Some(true);
        }
    }

    // Prove each target edge y->x with bound w by computing shortest paths from y.
    for te in target_edges {
        let src = *idx.get(&te.from)?;
        let dst = *idx.get(&te.to)?;
        // Bellman-Ford from src.
        let mut d = vec![i64::MAX / 4; n];
        d[src] = 0;
        for _ in 0..n.saturating_sub(1) {
            let mut changed = false;
            for (u, v, w) in edges.iter().copied() {
                if d[u] >= i64::MAX / 8 {
                    continue;
                }
                let cand = d[u].saturating_add(w);
                if cand < d[v] {
                    d[v] = cand;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // If we have a shortest-path bound from te.from to te.to that is <= te.w, the target holds.
        let bound = d[dst];
        if bound <= te.w {
            continue;
        } else {
            // Cannot prove this target edge from the partial encoding.
            return None;
        }
    }
    Some(true)
}

/// Entailment check on a `pp_dump`-shaped JSON payload:
/// UNSAT(hyps ∧ ¬target) => `Some(true)`
/// SAT(hyps ∧ ¬target)   => `Some(false)`
/// UNKNOWN / not-parsable => `None`
pub fn entails_from_pp_dump(
    pp_dump: &Value,
    timeout_ms: u64,
    seed: u64,
) -> Result<Option<bool>, String> {
    entails_from_pp_dump_with_depth(pp_dump, timeout_ms, seed, 0)
}

/// Like `entails_from_pp_dump_with_depth`, but reuses a “warm” solver session when available.
///
/// This is intended for callers that perform many SMT checks in a loop.
pub fn entails_from_pp_dump_with_depth_reuse(
    pp_dump: &Value,
    timeout_ms: u64,
    seed: u64,
    depth: usize,
    reuse: &mut Option<ReusableSmtSession>,
) -> Result<Option<bool>, String> {
    // Parse same as the non-reuse path.
    let goal = pp_dump
        .get("goals")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| "pp_dump missing goals[0]".to_string())?;

    let pretty = goal.get("pretty").and_then(|v| v.as_str()).unwrap_or("");
    let target = pretty
        .lines()
        .find_map(|ln| {
            ln.trim_start()
                .strip_prefix("⊢")
                .map(|r| r.trim().to_string())
        })
        .unwrap_or_default();
    if target.is_empty() {
        return Ok(None);
    }

    let mut var_kinds: std::collections::BTreeMap<String, VarKind> =
        std::collections::BTreeMap::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                if let Some((name, kind)) = extract_decl_kind(txt) {
                    var_kinds.insert(name, kind);
                }
            }
        }
    }

    let target_rel = match parse_rel_constraint_int(&target) {
        Some(r) => r,
        None => return Ok(None),
    };

    let mut hyp_rels: Vec<ParsedRelConstraint> = Vec::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                let rhs = txt
                    .split_once(':')
                    .map(|(_, r)| r.trim())
                    .unwrap_or(txt.trim());
                if rhs.is_empty() {
                    continue;
                }
                if let Some(r) = parse_rel_constraint_int(rhs) {
                    hyp_rels.push(r);
                }
            }
        }
    }
    let hyp_rels = select_constraints_by_var_depth(&target_rel.vars, &hyp_rels, depth);

    // Fast proofs before any solver use.
    if idl_proves_entails(&target_rel, &hyp_rels, &var_kinds) == Some(true)
        || entails_by_drop_nonneg_var(&target_rel.rel, &hyp_rels) == Some(true)
        || entails_by_offset_addition(&target_rel.rel, &hyp_rels) == Some(true)
    {
        return Ok(Some(true));
    }

    // Vars we actually need.
    let mut used_vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    used_vars.extend(target_rel.vars.iter().cloned());
    for r in &hyp_rels {
        used_vars.extend(r.vars.iter().cloned());
    }
    if used_vars.is_empty() {
        return Ok(None);
    }
    // Best-effort: if we can't recover a declared kind for a variable (common when the pretty
    // context doesn't include `x : Nat/Int` lines), default it to `Int` so we can still try SMT.
    // This is a heuristic signal only; unknown kinds should not block the search.
    for m in used_vars.iter() {
        var_kinds.entry(m.clone()).or_insert(VarKind::Int);
    }

    // Try the reused session (if available), otherwise initialize it.
    if reuse.is_none() {
        *reuse = ReusableSmtSession::new()?;
    }
    if let Some(sess) = reuse.as_mut() {
        // Best-effort: if reuse fails, disable it (but keep stats) and fall back to per-call spawning.
        let reused = (|| -> Result<Option<bool>, String> {
            if sess.disabled {
                return Ok(None);
            }
            sess.ensure_declared(&used_vars, &var_kinds)?;
            // Solver-specific strategy:
            // - Z3: push/pop + cached fragment is fast/stable.
            // - cvc5: prefer `check-sat-assuming` (avoids some stdout oddities), then fall back to push/pop.
            if sess.solver_used.trim_start().starts_with("cvc5") {
                if let Some(r) = sess.check_entails_assuming(
                    timeout_ms,
                    seed,
                    &hyp_rels,
                    &target_rel,
                    &used_vars,
                )? {
                    return Ok(Some(r));
                }
                sess.check_entails_pushpop(timeout_ms, seed, &hyp_rels, &target_rel, &used_vars)
            } else {
                // Prefer fragment caching + push/pop (smaller per-check traffic).
                if let Some(r) = sess.check_entails_pushpop(
                    timeout_ms,
                    seed,
                    &hyp_rels,
                    &target_rel,
                    &used_vars,
                )? {
                    return Ok(Some(r));
                }
                // Fallback to assuming-mode if push/pop path yields unknown.
                sess.check_entails_assuming(timeout_ms, seed, &hyp_rels, &target_rel, &used_vars)
            }
        })();
        match reused {
            Ok(Some(r)) => return Ok(Some(r)),
            Ok(None) => { /* continue to fallback */ }
            Err(e) => sess.disable(e),
        }
    }

    // Fallback: old behavior (spawn per call).
    entails_from_pp_dump_with_depth(pp_dump, timeout_ms, seed, depth)
}

/// Like `entails_from_pp_dump`, but optionally restricts which hypotheses are considered
/// using a bounded variable-connectivity expansion.
///
/// `depth` semantics:
/// - `0`: use all parseable LIA hypotheses (status quo)
/// - `1`: only hypotheses that mention a target variable
/// - `2+`: expand by shared variables up to `depth` passes
pub fn entails_from_pp_dump_with_depth(
    pp_dump: &Value,
    timeout_ms: u64,
    seed: u64,
    depth: usize,
) -> Result<Option<bool>, String> {
    use smtkit::smt2::t;

    let goal = pp_dump
        .get("goals")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| "pp_dump missing goals[0]".to_string())?;

    let pretty = goal.get("pretty").and_then(|v| v.as_str()).unwrap_or("");
    let target = pretty
        .lines()
        .find_map(|ln| {
            ln.trim_start()
                .strip_prefix("⊢")
                .map(|r| r.trim().to_string())
        })
        .unwrap_or_default();
    if target.is_empty() {
        return Ok(None);
    }

    let mut var_kinds: std::collections::BTreeMap<String, VarKind> =
        std::collections::BTreeMap::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                if let Some((name, kind)) = extract_decl_kind(txt) {
                    var_kinds.insert(name, kind);
                }
            }
        }
    }

    let target_rel = match parse_rel_constraint_int(&target) {
        Some(r) => r,
        None => return Ok(None),
    };

    let mut hyp_rels: Vec<ParsedRelConstraint> = Vec::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                let rhs = txt
                    .split_once(':')
                    .map(|(_, r)| r.trim())
                    .unwrap_or(txt.trim());
                if rhs.is_empty() {
                    continue;
                }
                if let Some(r) = parse_rel_constraint_int(rhs) {
                    hyp_rels.push(r);
                }
            }
        }
    }

    // Optionally restrict hyps by variable connectivity. This can materially reduce
    // SMT search time when there are many unrelated arithmetic hypotheses in scope.
    let hyp_rels = select_constraints_by_var_depth(&target_rel.vars, &hyp_rels, depth);

    // Fast path: try to *prove* entailment using difference-logic (IDL) when possible.
    // This is solver-free and avoids spawning an external process in common cases.
    if idl_proves_entails(&target_rel, &hyp_rels, &var_kinds) == Some(true) {
        return Ok(Some(true));
    }

    // We only need sorts/kinds for vars that appear in the selected fragment.
    let mut used_vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    used_vars.extend(target_rel.vars.iter().cloned());
    for r in &hyp_rels {
        used_vars.extend(r.vars.iter().cloned());
    }
    if used_vars.is_empty() {
        return Ok(None);
    }
    // Best-effort: default unknown variable kinds to `Int` so we can still attempt SMT.
    // This is a heuristic signal only; unknown kinds should not block the search.
    for m in used_vars.iter() {
        var_kinds.entry(m.clone()).or_insert(VarKind::Int);
    }

    let (mut sess, _used) = match smtkit::session::spawn_auto() {
        Ok(v) => v,
        Err(_) => {
            // Solver not available: fall back to cheap proofs only (best-effort).
            return Ok(idl_proves_entails(&target_rel, &hyp_rels, &var_kinds)
                .or_else(|| entails_by_drop_nonneg_var(&target_rel.rel, &hyp_rels))
                .or_else(|| entails_by_offset_addition(&target_rel.rel, &hyp_rels)));
        }
    };
    sess.set_logic("QF_LIA").map_err(|e| e.to_string())?;
    sess.set_print_success(false).map_err(|e| e.to_string())?;
    sess.set_produce_models(false).map_err(|e| e.to_string())?;
    sess.set_timeout_ms(timeout_ms).map_err(|e| e.to_string())?;
    sess.set_random_seed(seed).map_err(|e| e.to_string())?;

    // Declare only the vars we actually used in the selected fragment, to keep the
    // problem instance small and avoid failing on unrelated missing decls.
    for name in used_vars.iter() {
        let kind = *var_kinds.get(name).unwrap_or(&VarKind::Int);
        sess.declare_const(name, &smtkit::smt2::Sort::Int.to_smt2())
            .map_err(|e| e.to_string())?;
        if kind == VarKind::Nat {
            sess.assert_sexp(&t::ge(t::sym(name.clone()), t::int_lit(0)))
                .map_err(|e| e.to_string())?;
        }
    }
    for r in &hyp_rels {
        sess.assert_sexp(&r.sexp).map_err(|e| e.to_string())?;
    }
    // Prefer `check-sat-assuming` (temporary assertion) when available; it matches the
    // “assumptions / warmed solver” pattern and avoids polluting the assertion stack.
    let st = match sess.check_sat_assuming(&[t::not(target_rel.sexp.clone())]) {
        Ok(st) => st,
        Err(_) => {
            // Fallback for solvers that don't support it: assert then check.
            sess.assert_sexp(&t::not(target_rel.sexp.clone()))
                .map_err(|e| e.to_string())?;
            sess.check_sat().map_err(|e| e.to_string())?
        }
    };
    match st {
        smtkit::session::Status::Unsat => Ok(Some(true)),
        smtkit::session::Status::Sat => Ok(Some(false)),
        smtkit::session::Status::Unknown => {
            Ok(idl_proves_entails(&target_rel, &hyp_rels, &var_kinds)
                .or_else(|| entails_by_drop_nonneg_var(&target_rel.rel, &hyp_rels))
                .or_else(|| entails_by_offset_addition(&target_rel.rel, &hyp_rels)))
        }
    }
}

/// Cheap, bounded explanation of what the SMT/LIA checker *would* assert for a goal.
///
/// This does not run a solver. It’s meant for “why did SMT matter?” UX and debugging.
pub fn explain_fragment_from_pp_dump(
    pp_dump: &Value,
    depth: usize,
    max_hyps: usize,
) -> Option<Value> {
    let goal = pp_dump
        .get("goals")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())?;
    let pretty = goal.get("pretty").and_then(|v| v.as_str()).unwrap_or("");
    let target = pretty
        .lines()
        .find_map(|ln| {
            ln.trim_start()
                .strip_prefix("⊢")
                .map(|r| r.trim().to_string())
        })
        .unwrap_or_default();
    if target.is_empty() {
        return None;
    }
    let target_rel = parse_rel_constraint_int(&target)?;

    let mut hyp_rels: Vec<ParsedRelConstraint> = Vec::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                let rhs = txt
                    .split_once(':')
                    .map(|(_, r)| r.trim())
                    .unwrap_or(txt.trim());
                if rhs.is_empty() {
                    continue;
                }
                if let Some(r) = parse_rel_constraint_int(rhs) {
                    hyp_rels.push(r);
                }
            }
        }
    }
    let selected = select_constraints_by_var_depth(&target_rel.vars, &hyp_rels, depth);

    let mut used_vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    used_vars.extend(target_rel.vars.iter().cloned());
    for r in &selected {
        used_vars.extend(r.vars.iter().cloned());
    }

    let mut sample: Vec<String> = selected
        .iter()
        .take(max_hyps)
        .map(|r| r.src.clone())
        .collect();
    // Stable presentation.
    sample.sort();

    Some(serde_json::json!({
        "depth": depth,
        "target": target_rel.src,
        "selected_hyps_count": selected.len(),
        "selected_hyps_sample": sample,
        "used_vars_count": used_vars.len(),
        "used_vars_sample": used_vars.into_iter().take(24).collect::<Vec<_>>(),
    }))
}

/// Optional: compute a small unsat core for the selected fragment.
///
/// This is **expensive** and solver-dependent. It is intended for debugging/explanations only.
/// Inspired by Yurichev's discussion of unsat cores: you must explicitly enable the feature
/// and track/names assertions; otherwise the solver cannot report a core.
pub fn unsat_core_from_pp_dump(
    pp_dump: &Value,
    timeout_ms: u64,
    seed: u64,
    depth: usize,
    max_items: usize,
) -> Result<Option<Value>, String> {
    use smtkit::smt2::t;

    fn sanitize_smt_sym(s: &str) -> Option<String> {
        let raw = s.trim();
        if raw.is_empty() {
            return None;
        }
        let mut out = String::new();
        for ch in raw.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                out.push(ch);
            } else {
                out.push('_');
            }
        }
        if out.is_empty() {
            return None;
        }
        if out
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            out.insert_str(0, "h_");
        }
        Some(out)
    }

    fn select_pairs_by_var_depth(
        seed_vars: &std::collections::BTreeSet<String>,
        hyps: &[(ParsedRelConstraint, Option<String>)],
        depth: usize,
    ) -> Vec<(ParsedRelConstraint, Option<String>)> {
        if depth == 0 {
            return hyps.to_vec();
        }
        let mut included_vars = seed_vars.clone();
        let mut selected: Vec<(ParsedRelConstraint, Option<String>)> = Vec::new();
        let mut remaining: Vec<(ParsedRelConstraint, Option<String>)> = hyps.to_vec();
        for _ in 0..depth {
            if remaining.is_empty() {
                break;
            }
            let frontier = included_vars.clone();
            let mut next_remaining: Vec<(ParsedRelConstraint, Option<String>)> = Vec::new();
            let mut newly_included: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for (rel, nm) in remaining.into_iter() {
                if rel.vars.iter().any(|v| frontier.contains(v)) {
                    newly_included.extend(rel.vars.iter().cloned());
                    selected.push((rel, nm));
                } else {
                    next_remaining.push((rel, nm));
                }
            }
            included_vars.extend(newly_included);
            remaining = next_remaining;
        }
        selected
    }

    let goal = match pp_dump
        .get("goals")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
    {
        Some(g) => g,
        None => return Ok(None),
    };
    let pretty = goal.get("pretty").and_then(|v| v.as_str()).unwrap_or("");
    let target = pretty
        .lines()
        .find_map(|ln| {
            ln.trim_start()
                .strip_prefix("⊢")
                .map(|r| r.trim().to_string())
        })
        .unwrap_or_default();
    if target.is_empty() {
        return Ok(None);
    }
    let target_rel = match parse_rel_constraint_int(&target) {
        Some(r) => r,
        None => return Ok(None),
    };

    let mut hyp_pairs: Vec<(ParsedRelConstraint, Option<String>)> = Vec::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                let (name_hint, rhs) = if let Some((lhs, r)) = txt.split_once(':') {
                    let nm = lhs
                        .trim()
                        .split_whitespace()
                        .next()
                        .and_then(sanitize_smt_sym);
                    (nm, r.trim())
                } else {
                    (None, txt.trim())
                };
                if rhs.is_empty() {
                    continue;
                }
                if let Some(r) = parse_rel_constraint_int(rhs) {
                    hyp_pairs.push((r, name_hint));
                }
            }
        }
    }
    let hyp_pairs = select_pairs_by_var_depth(&target_rel.vars, &hyp_pairs, depth);

    // Vars we actually need.
    let mut used_vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    used_vars.extend(target_rel.vars.iter().cloned());
    for (r, _) in &hyp_pairs {
        used_vars.extend(r.vars.iter().cloned());
    }
    if used_vars.is_empty() {
        return Ok(None);
    }

    // Best-effort kind recovery (see entailment code).
    let mut var_kinds: std::collections::BTreeMap<String, VarKind> =
        std::collections::BTreeMap::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                if let Some((name, kind)) = extract_decl_kind(txt) {
                    var_kinds.insert(name, kind);
                }
            }
        }
    }
    for m in used_vars.iter() {
        var_kinds.entry(m.clone()).or_insert(VarKind::Int);
    }

    // Spawn a fresh session: unsat core production is often a global mode and can slow things down.
    let (mut sess, used) = match smtkit::session::spawn_auto() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    sess.set_logic("QF_LIA").map_err(|e| e.to_string())?;
    sess.set_print_success(false).map_err(|e| e.to_string())?;
    sess.set_produce_models(false).map_err(|e| e.to_string())?;
    sess.set_produce_unsat_cores(true)
        .map_err(|e| e.to_string())?;
    sess.set_timeout_ms(timeout_ms).map_err(|e| e.to_string())?;
    sess.set_random_seed(seed).map_err(|e| e.to_string())?;

    // Declare vars.
    for name in used_vars.iter() {
        let kind = *var_kinds.get(name).unwrap_or(&VarKind::Int);
        sess.declare_const(name, &smtkit::smt2::Sort::Int.to_smt2())
            .map_err(|e| e.to_string())?;
        if kind == VarKind::Nat {
            sess.assert_sexp(&t::ge(t::sym(name.clone()), t::int_lit(0)))
                .map_err(|e| e.to_string())?;
        }
    }

    // Assert hypotheses with names (SMT-LIB `:named`) so the core reports identifiers.
    //
    // Note: `:named` is widely supported (incl. Z3). If a solver doesn't support cores, it may
    // still return `unknown` or a core retrieval error, which we treat as `None`.
    let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut name_to_src: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    for (i, (h, hint)) in hyp_pairs.iter().enumerate() {
        let mut name = hint.clone().unwrap_or_else(|| format!("h{i}"));
        if used_names.contains(&name) {
            name = format!("{name}_{i}");
        }
        used_names.insert(name.clone());
        name_to_src.insert(name.clone(), h.src.clone());
        let term = t::app(
            "!",
            vec![
                h.sexp.clone(),
                smtkit::sexp::Sexp::atom(":named"),
                t::sym(name.clone()),
            ],
        );
        sess.assert_sexp(&term).map_err(|e| e.to_string())?;
    }
    // Name the negated target too (useful to see if it's in the core).
    name_to_src.insert("neg_target".to_string(), format!("¬({})", target_rel.src));
    let nt = t::app(
        "!",
        vec![
            t::not(target_rel.sexp.clone()),
            smtkit::sexp::Sexp::atom(":named"),
            t::sym("neg_target".to_string()),
        ],
    );
    sess.assert_sexp(&nt).map_err(|e| e.to_string())?;

    let st = sess.check_sat().map_err(|e| e.to_string())?;
    if st != smtkit::session::Status::Unsat {
        return Ok(None);
    }

    let core = sess.get_unsat_core().map_err(|e| e.to_string())?;
    let mut names: Vec<String> = Vec::new();
    if let smtkit::sexp::Sexp::List(items) = core {
        for it in items {
            if let smtkit::sexp::Sexp::Atom(a) = it {
                names.push(a);
            }
        }
    }
    if names.is_empty() {
        return Ok(None);
    }
    if names.len() > max_items {
        names.truncate(max_items);
    }
    let mut core_items: Vec<Value> = Vec::new();
    for nm in &names {
        if let Some(src) = name_to_src.get(nm) {
            core_items.push(serde_json::json!({ "name": nm, "src": src }));
        }
    }
    Ok(Some(serde_json::json!({
        "solver": used,
        "depth": depth,
        "core": names,
        "core_items": core_items,
    })))
}

/// Optional: capture an UNSAT proof object for the selected fragment (solver-dependent).
///
/// This is intended for **debugging/provenance**, not as a proof checker:
/// - the proof format is solver-specific
/// - the object may be large
/// - callers should keep it bounded and treat failure as "proof unavailable"
pub fn unsat_proof_from_pp_dump(
    pp_dump: &Value,
    timeout_ms: u64,
    seed: u64,
    depth: usize,
    max_chars: usize,
) -> Result<Option<Value>, String> {
    use smtkit::smt2::t;

    fn truncate_chars(s: &str, max: usize) -> String {
        if max == 0 {
            return String::new();
        }
        if s.chars().count() <= max {
            return s.to_string();
        }
        let mut out: String = s.chars().take(max).collect();
        out.push_str("…");
        out
    }

    let goal = match pp_dump
        .get("goals")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
    {
        Some(g) => g,
        None => return Ok(None),
    };
    let pretty = goal.get("pretty").and_then(|v| v.as_str()).unwrap_or("");
    let target = pretty
        .lines()
        .find_map(|ln| {
            ln.trim_start()
                .strip_prefix("⊢")
                .map(|r| r.trim().to_string())
        })
        .unwrap_or_default();
    if target.is_empty() {
        return Ok(None);
    }
    let target_rel = match parse_rel_constraint_int(&target) {
        Some(r) => r,
        None => return Ok(None),
    };

    // Collect parseable hypotheses, then (optionally) depth-filter.
    let mut hyp_rels: Vec<ParsedRelConstraint> = Vec::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                let rhs = txt
                    .split_once(':')
                    .map(|(_, r)| r.trim())
                    .unwrap_or(txt.trim());
                if rhs.is_empty() {
                    continue;
                }
                if let Some(r) = parse_rel_constraint_int(rhs) {
                    hyp_rels.push(r);
                }
            }
        }
    }
    let hyp_rels = select_constraints_by_var_depth(&target_rel.vars, &hyp_rels, depth);

    // Vars we actually need.
    let mut used_vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    used_vars.extend(target_rel.vars.iter().cloned());
    for r in &hyp_rels {
        used_vars.extend(r.vars.iter().cloned());
    }
    if used_vars.is_empty() {
        return Ok(None);
    }

    // Best-effort kind recovery.
    let mut var_kinds: std::collections::BTreeMap<String, VarKind> =
        std::collections::BTreeMap::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                if let Some((name, kind)) = extract_decl_kind(txt) {
                    var_kinds.insert(name, kind);
                }
            }
        }
    }
    for m in used_vars.iter() {
        var_kinds.entry(m.clone()).or_insert(VarKind::Int);
    }

    // Spawn a fresh session: proof production is often a global mode.
    let (mut sess, used) = match smtkit::session::spawn_auto() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    sess.set_logic("QF_LIA").map_err(|e| e.to_string())?;
    sess.set_print_success(false).map_err(|e| e.to_string())?;
    sess.set_produce_models(false).map_err(|e| e.to_string())?;
    sess.set_produce_proofs(true).map_err(|e| e.to_string())?;
    sess.set_timeout_ms(timeout_ms).map_err(|e| e.to_string())?;
    sess.set_random_seed(seed).map_err(|e| e.to_string())?;

    // Declare vars.
    for name in used_vars.iter() {
        let kind = *var_kinds.get(name).unwrap_or(&VarKind::Int);
        sess.declare_const(name, &smtkit::smt2::Sort::Int.to_smt2())
            .map_err(|e| e.to_string())?;
        if kind == VarKind::Nat {
            sess.assert_sexp(&t::ge(t::sym(name.clone()), t::int_lit(0)))
                .map_err(|e| e.to_string())?;
        }
    }

    // Assert hypotheses + negate target.
    for h in &hyp_rels {
        sess.assert_sexp(&h.sexp).map_err(|e| e.to_string())?;
    }
    sess.assert_sexp(&t::not(target_rel.sexp.clone()))
        .map_err(|e| e.to_string())?;

    let st = sess.check_sat().map_err(|e| e.to_string())?;
    if st != smtkit::session::Status::Unsat {
        return Ok(None);
    }

    let proof = sess.get_proof().map_err(|e| e.to_string())?;
    let proof_s = proof.to_string();
    let proof_chars = proof_s.chars().count();
    Ok(Some(serde_json::json!({
        "solver": used,
        "depth": depth,
        "chars": proof_chars,
        "preview": truncate_chars(&proof_s, max_chars),
    })))
}

/// Build an SMT-LIB2 script for the selected LIA fragment.
///
/// Intended for reproducible debugging outside proofpatch:
/// - you can run it with `z3 -in -smt2 < file.smt2`
/// - you can swap solvers and compare behavior/speed (a common workflow in Yurichev's examples)
pub fn smt2_script_from_pp_dump(
    pp_dump: &Value,
    timeout_ms: u64,
    seed: u64,
    depth: usize,
) -> Option<String> {
    use smtkit::smt2::t;

    fn sanitize_smt_sym(s: &str) -> Option<String> {
        let raw = s.trim();
        if raw.is_empty() {
            return None;
        }
        let mut out = String::new();
        for ch in raw.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                out.push(ch);
            } else {
                out.push('_');
            }
        }
        if out.is_empty() {
            return None;
        }
        if out
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            out.insert_str(0, "h_");
        }
        Some(out)
    }

    fn select_pairs_by_var_depth(
        seed_vars: &std::collections::BTreeSet<String>,
        hyps: &[(ParsedRelConstraint, Option<String>)],
        depth: usize,
    ) -> Vec<(ParsedRelConstraint, Option<String>)> {
        if depth == 0 {
            return hyps.to_vec();
        }
        let mut included_vars = seed_vars.clone();
        let mut selected: Vec<(ParsedRelConstraint, Option<String>)> = Vec::new();
        let mut remaining: Vec<(ParsedRelConstraint, Option<String>)> = hyps.to_vec();
        for _ in 0..depth {
            if remaining.is_empty() {
                break;
            }
            let frontier = included_vars.clone();
            let mut next_remaining: Vec<(ParsedRelConstraint, Option<String>)> = Vec::new();
            let mut newly_included: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for (rel, nm) in remaining.into_iter() {
                if rel.vars.iter().any(|v| frontier.contains(v)) {
                    newly_included.extend(rel.vars.iter().cloned());
                    selected.push((rel, nm));
                } else {
                    next_remaining.push((rel, nm));
                }
            }
            included_vars.extend(newly_included);
            remaining = next_remaining;
        }
        selected
    }

    let goal = pp_dump
        .get("goals")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())?;

    let pretty = goal.get("pretty").and_then(|v| v.as_str()).unwrap_or("");
    let target = pretty
        .lines()
        .find_map(|ln| {
            ln.trim_start()
                .strip_prefix("⊢")
                .map(|r| r.trim().to_string())
        })
        .unwrap_or_default();
    if target.is_empty() {
        return None;
    }
    let target_rel = parse_rel_constraint_int(&target)?;

    let mut var_kinds: std::collections::BTreeMap<String, VarKind> =
        std::collections::BTreeMap::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                if let Some((name, kind)) = extract_decl_kind(txt) {
                    var_kinds.insert(name, kind);
                }
            }
        }
    }

    let mut hyp_pairs: Vec<(ParsedRelConstraint, Option<String>)> = Vec::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                let (name_hint, rhs) = if let Some((lhs, r)) = txt.split_once(':') {
                    let nm = lhs
                        .trim()
                        .split_whitespace()
                        .next()
                        .and_then(sanitize_smt_sym);
                    (nm, r.trim())
                } else {
                    (None, txt.trim())
                };
                if rhs.is_empty() {
                    continue;
                }
                if let Some(r) = parse_rel_constraint_int(rhs) {
                    hyp_pairs.push((r, name_hint));
                }
            }
        }
    }
    let hyp_pairs = select_pairs_by_var_depth(&target_rel.vars, &hyp_pairs, depth);

    // Vars we actually need.
    let mut used_vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    used_vars.extend(target_rel.vars.iter().cloned());
    for (r, _) in &hyp_pairs {
        used_vars.extend(r.vars.iter().cloned());
    }
    if used_vars.is_empty() {
        return None;
    }
    for m in used_vars.iter() {
        var_kinds.entry(m.clone()).or_insert(VarKind::Int);
    }

    // Delegate SMT-LIB formatting to `smtkit` so `proofpatch` and other clients share a single,
    // canonical script surface.
    let mut script = smtkit::smt2::Script::new();
    script.comment(" proofpatch SMT fragment dump (LIA)");
    script.comment(" Run with: z3 -in -smt2 < this_file.smt2");
    script.set_option(":print-success", smtkit::sexp::Sexp::atom("false"));
    script.set_option(":produce-models", smtkit::sexp::Sexp::atom("false"));
    // For debugging/auditability, prefer a script that can produce cores/proofs if the solver supports them.
    script.set_option(":produce-unsat-cores", smtkit::sexp::Sexp::atom("true"));
    script.set_option(":produce-proofs", smtkit::sexp::Sexp::atom("true"));
    script.set_option(":timeout", smtkit::sexp::Sexp::atom(timeout_ms.to_string()));
    script.set_option(":random-seed", smtkit::sexp::Sexp::atom(seed.to_string()));
    script.set_logic("QF_LIA");

    for name in used_vars.iter() {
        script.declare_const(&smtkit::smt2::Var::new(
            name.clone(),
            smtkit::smt2::Sort::Int,
        ));
        let kind = *var_kinds.get(name).unwrap_or(&VarKind::Int);
        if kind == VarKind::Nat {
            script.assert(t::ge(t::sym(name.clone()), t::int_lit(0)));
        }
    }

    let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, (h, hint)) in hyp_pairs.iter().enumerate() {
        // Named assertions help when you later ask for a core.
        let mut nm = hint.clone().unwrap_or_else(|| format!("h{i}"));
        if used_names.contains(&nm) {
            nm = format!("{nm}_{i}");
        }
        used_names.insert(nm.clone());
        let named = t::app(
            "!",
            vec![
                h.sexp.clone(),
                smtkit::sexp::Sexp::atom(":named"),
                t::sym(nm),
            ],
        );
        script.assert(named);
    }
    let nt = t::app(
        "!",
        vec![
            t::not(target_rel.sexp.clone()),
            smtkit::sexp::Sexp::atom(":named"),
            t::sym("neg_target".to_string()),
        ],
    );
    script.assert(nt);
    script.check_sat();
    script.comment(" If UNSAT, you can fetch artifacts by uncommenting:");
    script.comment(" (get-unsat-core)");
    script.comment(" (get-proof)");
    script.comment(" (exit)");
    Some(script.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smt_unsat_proof_capture_smoke() {
        // This test is best-effort and intentionally *skips* if no solver is available.
        // It validates the end-to-end `(get-proof)` plumbing for a trivial entailment:
        //   h : x ≤ y  ⊢  x ≤ y
        //
        // We model entailment as UNSAT of: (hyps ∧ ¬target).

        // Skip if `smtkit` can't spawn a solver.
        if smtkit::session::spawn_auto().is_err() {
            return;
        }

        let pp_dump = serde_json::json!({
            "goals": [{
                "pretty": "x y : Int\nh : x ≤ y\n⊢ x ≤ y",
                "hyps": [
                    { "text": "x : Int" },
                    { "text": "y : Int" },
                    { "text": "h : x ≤ y" }
                ]
            }]
        });

        let pf = unsat_proof_from_pp_dump(&pp_dump, 5_000, 0, 0, 2_000)
            .expect("unsat_proof_from_pp_dump should not error")
            .expect("expected Some(proof) for trivial UNSAT");

        let preview = pf.get("preview").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            preview.contains("proof"),
            "expected proof preview to mention 'proof', got: {preview}"
        );
    }

    #[test]
    fn smt_depth_selects_connected_constraints() {
        let mk = |s: &str| parse_rel_constraint_int(s).expect("parse");
        let target = mk("a <= 0");
        let h1 = mk("a <= b");
        let h2 = mk("b <= c");
        let h3 = mk("d <= e");
        let hyps = vec![h1.clone(), h2.clone(), h3.clone()];

        // depth=0 keeps status quo (all hyps).
        assert_eq!(
            select_constraints_by_var_depth(&target.vars, &hyps, 0).len(),
            3
        );
        // depth=1: only direct mentions of `a`.
        assert_eq!(
            select_constraints_by_var_depth(&target.vars, &hyps, 1).len(),
            1
        );
        // depth=2: bring in `b <= c` via `a <= b`.
        assert_eq!(
            select_constraints_by_var_depth(&target.vars, &hyps, 2).len(),
            2
        );
    }

    #[test]
    fn idl_proves_transitive_entailment() {
        // a <= b + 1, b <= c + 2  =>  a <= c + 3
        let mk = |s: &str| parse_rel_constraint_int(s).expect("parse");
        let h1 = mk("a <= b + 1");
        let h2 = mk("b <= c + 2");
        let target = mk("a <= c + 3");
        let mut kinds = std::collections::BTreeMap::new();
        kinds.insert("a".to_string(), VarKind::Int);
        kinds.insert("b".to_string(), VarKind::Int);
        kinds.insert("c".to_string(), VarKind::Int);
        assert_eq!(idl_proves_entails(&target, &[h1, h2], &kinds), Some(true));
    }
}
