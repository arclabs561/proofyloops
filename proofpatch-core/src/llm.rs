use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResult {
    pub provider: String,
    pub model: String,
    /// Where the model string came from (env vs default).
    pub model_source: String,
    /// Which env var was used for the provider's model (when applicable).
    pub model_env: String,
    pub content: String,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedProviderInfo {
    pub provider: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub model_source: String,
    pub model_env: String,
}

#[derive(Debug, Clone)]
struct Provider {
    name: &'static str,
    base_url: String,
    api_key_env: Option<&'static str>,
    model_env: &'static str,
}

fn providers_from_env() -> Vec<Provider> {
    vec![
        Provider {
            name: "ollama",
            base_url: std::env::var("OLLAMA_HOST")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "http://localhost:11434".to_string())
                .trim_end_matches('/')
                .to_string(),
            api_key_env: None,
            model_env: "OLLAMA_MODEL",
        },
        Provider {
            name: "groq",
            base_url: "https://api.groq.com/openai/v1".to_string(),
            api_key_env: Some("GROQ_API_KEY"),
            model_env: "GROQ_MODEL",
        },
        Provider {
            name: "openrouter",
            base_url: std::env::var("OPENROUTER_BASE_URL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string())
                .trim_end_matches('/')
                .to_string(),
            api_key_env: Some("OPENROUTER_API_KEY"),
            model_env: "OPENROUTER_MODEL",
        },
        // Not in the original Python router, but used by covolume's llm_review.
        Provider {
            name: "openai",
            base_url: std::env::var("OPENAI_BASE_URL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
                .trim_end_matches('/')
                .to_string(),
            api_key_env: Some("OPENAI_API_KEY"),
            model_env: "OPENAI_MODEL",
        },
    ]
}

fn provider_order() -> Vec<String> {
    for k in ["PROOFPATCH_PROVIDER_ORDER"] {
        if let Ok(v) = std::env::var(k) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return v
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }
    vec![
        "ollama".into(),
        "groq".into(),
        "openai".into(),
        "openrouter".into(),
    ]
}

fn env_nonempty(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .as_deref()
        .unwrap_or("")
        .trim()
        .len()
        > 0
}

fn env_truthy(name: &str, default_on: bool) -> bool {
    let v = std::env::var(name).ok().unwrap_or_default();
    let v = v.trim().to_lowercase();
    if v.is_empty() {
        return default_on;
    }
    !matches!(v.as_str(), "0" | "false" | "no" | "off")
}

fn default_model_from_env(provider_name: &str) -> Option<(String, String)> {
    // Provider-specific override wins.
    let key1 = format!("PROOFPATCH_DEFAULT_MODEL_{}", provider_name.to_uppercase());
    for k in [
        key1.as_str(),
        "PROOFPATCH_DEFAULT_MODEL",
    ]
    {
        if let Ok(v) = std::env::var(k) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some((v, k.to_string()));
            }
        }
    }
    None
}

fn hardcoded_default_model(provider_name: &str) -> Option<&'static str> {
    // Keep these conservative and stable-ish. Users can override with env vars above.
    // Note: we intentionally do NOT pick a default for Ollama; local installs vary.
    match provider_name {
        // Prefer a high-capability default on OpenRouter. Override with OPENROUTER_MODEL.
        //
        // Rationale: OpenRouter's own usage leaderboard strongly favors Claude 4.5 variants.
        // See: https://openrouter.ai/rankings
        "openrouter" => Some("anthropic/claude-opus-4.5"),
        "openai" => Some("gpt-4o-mini"),
        "groq" => Some("llama-3.1-8b-instant"),
        _ => None,
    }
}

async fn is_ollama_reachable(base_url: &str, timeout: Duration) -> bool {
    // Match the Python client: try /v1/models first, then /api/tags.
    let client = match reqwest::Client::builder().timeout(timeout).build() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let candidates = [
        format!("{base_url}/v1/models"),
        format!("{base_url}/api/tags"),
    ];
    for u in candidates {
        let r = client.get(u).send().await;
        match r {
            Ok(resp) => {
                let status = resp.status().as_u16();
                // 200 OK, 401 unauthorized, 404 not found all indicate "there is something there".
                if matches!(status, 200 | 401 | 404) {
                    return true;
                }
            }
            Err(_) => continue,
        }
    }
    false
}

async fn select_provider(timeout: Duration) -> Result<(Provider, String, String), String> {
    let provs = providers_from_env();
    for name in provider_order() {
        let Some(p) = provs.iter().find(|pp| pp.name == name).cloned() else {
            continue;
        };
        let mut model = std::env::var(p.model_env).ok().unwrap_or_default();
        model = model.trim().to_string();
        if let Some(k) = p.api_key_env {
            if !env_nonempty(k) {
                continue;
            }
        }
        let mut model_source = "env".to_string();
        if model.is_empty() {
            // Defaults are enabled unless explicitly disabled.
            let defaults_enabled = env_truthy("PROOFPATCH_MODEL_DEFAULTS", true);
            if !defaults_enabled {
                continue;
            }
            if let Some((m, key)) = default_model_from_env(p.name) {
                model = m;
                model_source = format!("default_env_override({})", key);
            } else if let Some(m) = hardcoded_default_model(p.name) {
                model = m.to_string();
                model_source = "default_hardcoded".to_string();
            } else {
                model = String::new();
            }
            if model.is_empty() {
                continue;
            }
        }
        if p.name == "ollama" && !is_ollama_reachable(&p.base_url, timeout).await {
            continue;
        }
        return Ok((p, model, model_source));
    }
    Err("No usable provider found. Set one of:\n\
- OLLAMA_MODEL (+ optional OLLAMA_HOST)\n\
- GROQ_API_KEY and GROQ_MODEL\n\
- OPENROUTER_API_KEY and OPENROUTER_MODEL\n\
- OPENAI_API_KEY and OPENAI_MODEL\n\
Optionally set:\n\
- PROOFPATCH_PROVIDER_ORDER\n\
- PROOFPATCH_DEFAULT_MODEL / PROOFPATCH_DEFAULT_MODEL_<PROVIDER>\n\
- PROOFPATCH_MODEL_DEFAULTS=0 to disable built-in defaults"
        .to_string())
}

pub async fn select_provider_info(timeout: Duration) -> Result<SelectedProviderInfo, String> {
    let (p, model, model_source) = select_provider(timeout).await?;
    let api_key = p.api_key_env.and_then(|k| {
        std::env::var(k)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    });
    Ok(SelectedProviderInfo {
        provider: p.name.to_string(),
        base_url: p.base_url,
        api_key,
        model,
        model_source,
        model_env: p.model_env.to_string(),
    })
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    message: Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

/// OpenAI-compatible chat completions, with provider selection matching the legacy Python CLI.
///
/// Invariants (should not change lightly):
/// - request path is `POST <base_url>/chat/completions`
/// - uses `Authorization: Bearer <key>` when provider requires a key
/// - OpenRouter adds `HTTP-Referer` and `X-Title` when configured
/// - default temperature is 0.2
pub async fn chat_completion(
    system: &str,
    user: &str,
    timeout: Duration,
) -> Result<ChatCompletionResult, String> {
    let (provider, model, model_source) = select_provider(Duration::from_secs(3)).await?;

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    if let Some(k) = provider.api_key_env {
        let key = std::env::var(k).ok().unwrap_or_default();
        let key = key.trim().to_string();
        if key.is_empty() {
            return Err(format!("missing {}", k));
        }
        let v = format!("Bearer {}", key);
        let hv = reqwest::header::HeaderValue::from_str(&v)
            .map_err(|e| format!("invalid Authorization header: {e}"))?;
        headers.insert(reqwest::header::AUTHORIZATION, hv);
    }
    if provider.name == "openrouter" {
        if let Ok(site) = std::env::var("OPENROUTER_SITE_URL") {
            let site = site.trim().to_string();
            if !site.is_empty() {
                if let Ok(hv) = reqwest::header::HeaderValue::from_str(&site) {
                    headers.insert("HTTP-Referer", hv);
                }
            }
        }
        if let Ok(app) = std::env::var("OPENROUTER_APP_NAME") {
            let app = app.trim().to_string();
            if !app.is_empty() {
                if let Ok(hv) = reqwest::header::HeaderValue::from_str(&app) {
                    headers.insert("X-Title", hv);
                }
            }
        }
    }

    let payload = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ],
        "temperature": 0.2
    });

    let url = format!("{}/chat/completions", provider.base_url);
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .default_headers(headers)
        .build()
        .map_err(|e| format!("http client build: {e}"))?;
    let resp = client
        .post(url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("http request failed: {e}"))?;

    let status = resp.status();
    let raw: Value = resp
        .json()
        .await
        .map_err(|e| format!("http json decode: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "provider {} returned {}: {}",
            provider.name,
            status.as_u16(),
            raw
        ));
    }

    let parsed: ChatCompletionResponse =
        serde_json::from_value(raw.clone()).map_err(|e| format!("invalid chat response: {e}"))?;
    let msg = parsed
        .choices
        .get(0)
        .and_then(|c| c.message.as_object())
        .cloned()
        .ok_or_else(|| "missing choices[0].message".to_string())?;
    let content = msg
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(ChatCompletionResult {
        provider: provider.name.to_string(),
        model,
        model_source,
        model_env: provider.model_env.to_string(),
        content,
        raw,
    })
}

/// Lower-level entrypoint: send an OpenAI-compatible request with an explicit `messages` array and optional `tools`.
///
/// Returns the raw JSON response plus injected `provider`/`model` fields for callers that want tool loops.
pub async fn chat_completion_raw(
    messages: &[serde_json::Value],
    tools: Option<&serde_json::Value>,
    timeout: Duration,
) -> Result<serde_json::Value, String> {
    let (provider, model, model_source) = select_provider(Duration::from_secs(3)).await?;

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    if let Some(k) = provider.api_key_env {
        let key = std::env::var(k).ok().unwrap_or_default();
        let key = key.trim().to_string();
        if key.is_empty() {
            return Err(format!("missing {}", k));
        }
        let v = format!("Bearer {}", key);
        let hv = reqwest::header::HeaderValue::from_str(&v)
            .map_err(|e| format!("invalid Authorization header: {e}"))?;
        headers.insert(reqwest::header::AUTHORIZATION, hv);
    }
    if provider.name == "openrouter" {
        if let Ok(site) = std::env::var("OPENROUTER_SITE_URL") {
            let site = site.trim().to_string();
            if !site.is_empty() {
                if let Ok(hv) = reqwest::header::HeaderValue::from_str(&site) {
                    headers.insert("HTTP-Referer", hv);
                }
            }
        }
        if let Ok(app) = std::env::var("OPENROUTER_APP_NAME") {
            let app = app.trim().to_string();
            if !app.is_empty() {
                if let Ok(hv) = reqwest::header::HeaderValue::from_str(&app) {
                    headers.insert("X-Title", hv);
                }
            }
        }
    }

    let mut payload = serde_json::json!({
        "model": model,
        "messages": messages,
        "temperature": 0.2
    });
    if let Some(t) = tools {
        payload["tools"] = t.clone();
        payload["tool_choice"] = serde_json::json!("auto");
    }

    let url = format!("{}/chat/completions", provider.base_url);
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .default_headers(headers)
        .build()
        .map_err(|e| format!("http client build: {e}"))?;
    let resp = client
        .post(url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("http request failed: {e}"))?;

    let status = resp.status();
    let mut raw: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("http json decode: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "provider {} returned {}: {}",
            provider.name,
            status.as_u16(),
            raw
        ));
    }
    // Inject selection metadata so callers don't have to re-run selection.
    if let Some(obj) = raw.as_object_mut() {
        obj.insert("provider".to_string(), serde_json::Value::String(provider.name.to_string()));
        obj.insert("model".to_string(), serde_json::Value::String(model));
        obj.insert(
            "model_source".to_string(),
            serde_json::Value::String(model_source),
        );
        obj.insert(
            "model_env".to_string(),
            serde_json::Value::String(provider.model_env.to_string()),
        );
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&str]) -> Self {
            let mut saved = Vec::new();
            for k in keys {
                saved.push((k.to_string(), std::env::var(k).ok()));
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in self.saved.drain(..) {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[test]
    fn provider_order_default() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFPATCH_PROVIDER_ORDER",
        ]);
        std::env::remove_var("PROOFPATCH_PROVIDER_ORDER");
        assert_eq!(
            provider_order(),
            vec![
                "ollama".to_string(),
                "groq".to_string(),
                "openai".to_string(),
                "openrouter".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn select_provider_openrouter_by_env() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFPATCH_PROVIDER_ORDER",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "OLLAMA_MODEL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        ]);
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_MODEL");
        std::env::remove_var("GROQ_API_KEY");
        std::env::remove_var("GROQ_MODEL");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::set_var("PROOFPATCH_PROVIDER_ORDER", "openrouter");
        std::env::set_var("OPENROUTER_API_KEY", "test_key");
        std::env::set_var("OPENROUTER_MODEL", "openai/gpt-4o-mini");

        let (p, model, src) = select_provider(Duration::from_millis(10))
            .await
            .expect("expected provider");
        assert_eq!(p.name, "openrouter");
        assert_eq!(model, "openai/gpt-4o-mini");
        assert_eq!(src, "env");
    }

    #[tokio::test]
    async fn select_provider_openrouter_uses_default_model_when_missing() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFPATCH_PROVIDER_ORDER",
            "PROOFPATCH_MODEL_DEFAULTS",
            "PROOFPATCH_DEFAULT_MODEL_OPENROUTER",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "OLLAMA_MODEL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        ]);
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_MODEL");
        std::env::remove_var("GROQ_API_KEY");
        std::env::remove_var("GROQ_MODEL");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("PROOFPATCH_DEFAULT_MODEL_OPENROUTER");
        std::env::set_var("PROOFPATCH_PROVIDER_ORDER", "openrouter");
        std::env::set_var("OPENROUTER_API_KEY", "test_key");
        // No OPENROUTER_MODEL set: should use hardcoded default.

        let (p, model, src) = select_provider(Duration::from_millis(10))
            .await
            .expect("expected provider");
        assert_eq!(p.name, "openrouter");
        assert_eq!(model, "anthropic/claude-opus-4.5");
        assert_eq!(src, "default_hardcoded");
    }

    #[tokio::test]
    async fn select_provider_defaults_can_be_disabled() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFPATCH_PROVIDER_ORDER",
            "PROOFPATCH_MODEL_DEFAULTS",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "OLLAMA_MODEL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        ]);
        std::env::set_var("PROOFPATCH_PROVIDER_ORDER", "openrouter");
        std::env::set_var("PROOFPATCH_MODEL_DEFAULTS", "0");
        std::env::set_var("OPENROUTER_API_KEY", "test_key");
        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("GROQ_API_KEY");
        std::env::remove_var("GROQ_MODEL");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_MODEL");

        let err = select_provider(Duration::from_millis(10))
            .await
            .unwrap_err();
        assert!(err.contains("No usable provider found"));
    }

    #[tokio::test]
    async fn select_provider_openai_by_env() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFPATCH_PROVIDER_ORDER",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "OLLAMA_MODEL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        ]);
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("GROQ_API_KEY");
        std::env::remove_var("GROQ_MODEL");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::set_var("PROOFPATCH_PROVIDER_ORDER", "openai");
        std::env::set_var("OPENAI_API_KEY", "test_key");
        std::env::set_var("OPENAI_MODEL", "gpt-4o-mini");

        let (p, model, src) = select_provider(Duration::from_millis(10))
            .await
            .expect("expected provider");
        assert_eq!(p.name, "openai");
        assert_eq!(model, "gpt-4o-mini");
        assert_eq!(src, "env");
    }

    #[tokio::test]
    async fn select_provider_errors_when_unconfigured() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFPATCH_PROVIDER_ORDER",
            "OLLAMA_MODEL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        ]);
        std::env::set_var(
            "PROOFPATCH_PROVIDER_ORDER",
            "openrouter,openai,groq,ollama",
        );
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("GROQ_API_KEY");
        std::env::remove_var("GROQ_MODEL");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_MODEL");

        let err = select_provider(Duration::from_millis(10))
            .await
            .unwrap_err();
        assert!(err.contains("No usable provider found"));
    }
}
