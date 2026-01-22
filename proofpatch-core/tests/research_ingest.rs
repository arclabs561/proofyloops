use proofpatch_core::ingest_research_json;

#[test]
fn ingest_dedupes_urls_and_keeps_titles() {
    let v = serde_json::json!({
        "results": [
            {"title":"A","url":"https://example.com/a","content":"x"},
            {"title":"A-dup","url":"https://example.com/a","content":"y"},
            {"title":"B","link":"https://example.com/b","snippet":"z"},
            {"title":"not-a-url","url":"file:///tmp/nope"},
            {"title":"schema-noise","url":"https://json-schema.org/draft-07/schema#"}
        ]
    });

    let notes = ingest_research_json(&v);
    assert_eq!(notes.deduped_urls, 2);
    assert!(notes.raw_urls >= 3);

    let mut urls: Vec<String> = notes.sources.iter().map(|s| s.url.clone()).collect();
    urls.sort();
    assert_eq!(urls, vec!["https://example.com/a", "https://example.com/b"]);

    let a = notes
        .sources
        .iter()
        .find(|s| s.url == "https://example.com/a")
        .expect("missing a");
    assert_eq!(a.title.as_deref(), Some("A"));
}

#[test]
fn ingest_propagates_origin_and_canonicalizes_arxiv_pdf() {
    let v = serde_json::json!({
        "tool_outputs": [
            {
                "tool": "tavily_search",
                "results": [
                    {"title":"Some PDF","url":"https://example.com/paper.pdf","snippet":"mentions Cauchy lemma"},
                    {"title":"Duplicate","url":"https://example.com/paper.pdf"}
                ]
            },
            {
                "tool": "arxiv",
                "papers": [
                    {"title":"On polygonal numbers","link":"https://arxiv.org/abs/1234.5678","abstract":"..."},
                    {"title":"On polygonal numbers (pdf)","pdf_url":"https://arxiv.org/pdf/1234.5678.pdf"}
                ]
            }
        ]
    });

    let notes = ingest_research_json(&v);

    // pdf + abs should dedupe to one canonical paper URL.
    assert_eq!(notes.deduped_urls, 2);

    let tav = notes
        .sources
        .iter()
        .find(|s| s.url == "https://example.com/paper.pdf")
        .expect("missing tavily source");
    assert_eq!(tav.origin.as_deref(), Some("tavily_search"));

    let arxiv_abs = notes
        .sources
        .iter()
        .find(|s| s.url.starts_with("https://arxiv.org/"))
        .expect("missing arxiv source");
    assert_eq!(arxiv_abs.canonical_url.as_deref(), Some("https://arxiv.org/abs/1234.5678"));
    assert_eq!(arxiv_abs.origin.as_deref(), Some("arxiv"));
}

