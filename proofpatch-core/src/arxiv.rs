use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArxivPaper {
    pub title: String,
    pub link: String,
    #[serde(default)]
    pub pdf_url: Option<String>,
    #[serde(default)]
    pub published: Option<String>,
    #[serde(default)]
    pub updated: Option<String>,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub abstract_text: String,
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_between(haystack: &str, start: &str, end: &str) -> Option<String> {
    let i = haystack.find(start)? + start.len();
    let rest = &haystack[i..];
    let j = rest.find(end)?;
    Some(rest[..j].to_string())
}

fn extract_first_tag_text(entry: &str, tag: &str) -> Option<String> {
    // Try `<tag>...</tag>` first, then `<tag ...>...</tag>`.
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    if let Some(v) = extract_between(entry, &open, &close) {
        return Some(normalize_ws(v.trim()));
    }
    let open2 = format!("<{tag} ");
    if let Some(i0) = entry.find(&open2) {
        let rest = &entry[i0..];
        let i1 = rest.find('>')? + 1;
        let rest2 = &rest[i1..];
        let j = rest2.find(&close)?;
        return Some(normalize_ws(rest2[..j].trim()));
    }
    None
}

fn extract_authors(entry: &str) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in entry.split("<author").skip(1) {
        let Some(block) = chunk.split("</author>").next() else {
            continue;
        };
        if let Some(name) = extract_first_tag_text(block, "name") {
            if !name.trim().is_empty() {
                out.push(name.trim().to_string());
            }
        }
    }
    out
}

fn extract_pdf_url(entry: &str) -> Option<String> {
    // Atom feed has multiple <link ...> elements; try to find one that looks like a PDF.
    for chunk in entry.split("<link").skip(1) {
        let Some(tag) = chunk.split('>').next() else {
            continue;
        };
        let tag_l = tag.to_lowercase();
        if !(tag_l.contains("pdf") || tag_l.contains("application/pdf")) {
            continue;
        }
        if let Some(href) = extract_between(tag, "href=\"", "\"") {
            if href.contains("arxiv.org") {
                return Some(href);
            }
        }
    }
    None
}

pub fn parse_arxiv_atom(xml: &str, max_results: usize) -> Vec<ArxivPaper> {
    let mut out = Vec::new();
    for entry_chunk in xml.split("<entry").skip(1) {
        let Some(entry_block0) = entry_chunk.split("</entry>").next() else {
            continue;
        };
        // Skip feed-level title/metadata; we only parse inside entries.
        let title = extract_first_tag_text(entry_block0, "title").unwrap_or_default();
        let link = extract_first_tag_text(entry_block0, "id").unwrap_or_default();
        if title.is_empty() || link.is_empty() {
            continue;
        }
        let published = extract_first_tag_text(entry_block0, "published");
        let updated = extract_first_tag_text(entry_block0, "updated");
        let abstract_text = extract_first_tag_text(entry_block0, "summary").unwrap_or_default();
        let authors = extract_authors(entry_block0);
        let pdf_url = extract_pdf_url(entry_block0);
        out.push(ArxivPaper {
            title,
            link,
            pdf_url,
            published,
            updated,
            authors,
            abstract_text,
        });
        if out.len() >= max_results {
            break;
        }
    }
    out
}

pub async fn arxiv_search(
    query: &str,
    max_results: usize,
    timeout: Duration,
) -> Result<Vec<ArxivPaper>, String> {
    let max_results = max_results.clamp(1, 50);
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| format!("reqwest client: {e}"))?;
    let mut url = reqwest::Url::parse("https://export.arxiv.org/api/query")
        .map_err(|e| format!("parse arxiv url: {e}"))?;
    url.query_pairs_mut()
        .append_pair("search_query", &format!("all:{query}"))
        .append_pair("start", "0")
        .append_pair("max_results", &max_results.to_string());
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("arxiv fetch: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("arxiv fetch status: {}", resp.status()));
    }
    let xml = resp.text().await.map_err(|e| format!("arxiv text: {e}"))?;
    Ok(parse_arxiv_atom(&xml, max_results))
}

