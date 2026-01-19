use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResult {
    pub provider: String,
    pub model: String,
    pub content: String,
    pub raw: Value,
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
        // Not in the original proofyloops Python router, but used by covolume's llm_review.
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
    // Preserve legacy env fallbacks (proofyloops/proofloops/leanpot).
    for k in ["PROOFYLOOPS_PROVIDER_ORDER", "PROOFLOOPS_PROVIDER_ORDER", "LEANPOT_PROVIDER_ORDER"] {
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
    vec!["ollama".into(), "groq".into(), "openrouter".into()]
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
 
async fn is_ollama_reachable(base_url: &str, timeout: Duration) -> bool {
    // Match the Python client: try /v1/models first, then /api/tags.
    let client = match reqwest::Client::builder().timeout(timeout).build() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let candidates = [format!("{base_url}/v1/models"), format!("{base_url}/api/tags")];
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
 
async fn select_provider(timeout: Duration) -> Result<(Provider, String), String> {
    let provs = providers_from_env();
    for name in provider_order() {
        let Some(p) = provs.iter().find(|pp| pp.name == name).cloned() else {
            continue;
        };
        let model = std::env::var(p.model_env).ok().unwrap_or_default();
        let model = model.trim().to_string();
        if model.is_empty() {
            continue;
        }
        if let Some(k) = p.api_key_env {
            if !env_nonempty(k) {
                continue;
            }
        }
        if p.name == "ollama" && !is_ollama_reachable(&p.base_url, timeout).await {
            continue;
        }
        return Ok((p, model));
    }
    Err(
        "No usable provider found. Set one of:\n\
- OLLAMA_MODEL (+ optional OLLAMA_HOST)\n\
- GROQ_API_KEY and GROQ_MODEL\n\
- OPENROUTER_API_KEY and OPENROUTER_MODEL\n\
- OPENAI_API_KEY and OPENAI_MODEL\n\
Optionally set PROOFYLOOPS_PROVIDER_ORDER."
            .to_string(),
    )
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
pub async fn chat_completion(system: &str, user: &str, timeout: Duration) -> Result<ChatCompletionResult, String> {
    let (provider, model) = select_provider(Duration::from_secs(3)).await?;
 
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
    let raw: Value = resp.json().await.map_err(|e| format!("http json decode: {e}"))?;
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
        content,
        raw,
    })
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
            "PROOFYLOOPS_PROVIDER_ORDER",
            "PROOFLOOPS_PROVIDER_ORDER",
            "LEANPOT_PROVIDER_ORDER",
        ]);
        std::env::remove_var("PROOFYLOOPS_PROVIDER_ORDER");
        std::env::remove_var("PROOFLOOPS_PROVIDER_ORDER");
        std::env::remove_var("LEANPOT_PROVIDER_ORDER");
        assert_eq!(
            provider_order(),
            vec!["ollama".to_string(), "groq".to_string(), "openrouter".to_string()]
        );
    }

    #[tokio::test]
    async fn select_provider_openrouter_by_env() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFYLOOPS_PROVIDER_ORDER",
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
        std::env::set_var("PROOFYLOOPS_PROVIDER_ORDER", "openrouter");
        std::env::set_var("OPENROUTER_API_KEY", "test_key");
        std::env::set_var("OPENROUTER_MODEL", "openai/gpt-4o-mini");

        let (p, model) = select_provider(Duration::from_millis(10))
            .await
            .expect("expected provider");
        assert_eq!(p.name, "openrouter");
        assert_eq!(model, "openai/gpt-4o-mini");
    }

    #[tokio::test]
    async fn select_provider_openai_by_env() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFYLOOPS_PROVIDER_ORDER",
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
        std::env::set_var("PROOFYLOOPS_PROVIDER_ORDER", "openai");
        std::env::set_var("OPENAI_API_KEY", "test_key");
        std::env::set_var("OPENAI_MODEL", "gpt-4o-mini");

        let (p, model) = select_provider(Duration::from_millis(10))
            .await
            .expect("expected provider");
        assert_eq!(p.name, "openai");
        assert_eq!(model, "gpt-4o-mini");
    }

    #[tokio::test]
    async fn select_provider_errors_when_unconfigured() {
        let _lock = env_lock().lock().unwrap();
        let _g = EnvGuard::new(&[
            "PROOFYLOOPS_PROVIDER_ORDER",
            "OLLAMA_MODEL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
        ]);
        std::env::set_var("PROOFYLOOPS_PROVIDER_ORDER", "openrouter,openai,groq,ollama");
        std::env::remove_var("OLLAMA_MODEL");
        std::env::remove_var("GROQ_API_KEY");
        std::env::remove_var("GROQ_MODEL");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("OPENAI_MODEL");

        let err = select_provider(Duration::from_millis(10)).await.unwrap_err();
        assert!(err.contains("No usable provider found"));
    }
}

