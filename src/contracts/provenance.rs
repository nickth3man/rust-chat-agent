use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceEntry {
    pub source_id: String,
    /// Canonical URL key used for cross-provider deduplication.
    pub normalized_url: String,
    pub title: String,
    pub url: String,
    pub supporting_snippet: String,
    pub rank_decision: Option<String>,
    #[serde(default)]
    pub provider_labels: BTreeSet<String>,
    #[serde(default)]
    pub source_subtypes: BTreeSet<String>,
}

impl EvidenceEntry {
    pub fn merge(&mut self, other: Self) {
        if self.title.is_empty() {
            self.title = other.title;
        }
        if self.url.is_empty() {
            self.url = other.url;
        }
        if self.supporting_snippet.is_empty() {
            self.supporting_snippet = other.supporting_snippet;
        }
        if self.rank_decision.is_none() {
            self.rank_decision = other.rank_decision;
        }
        self.provider_labels.extend(other.provider_labels);
        self.source_subtypes.extend(other.source_subtypes);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TurnProvenance {
    pub turn_id: String,
    #[serde(default)]
    pub entries: Vec<EvidenceEntry>,
}

impl TurnProvenance {
    pub fn merge(&mut self, incoming: impl IntoIterator<Item = EvidenceEntry>) {
        for entry in incoming {
            if let Some(existing) = self
                .entries
                .iter_mut()
                .find(|item| item.normalized_url == entry.normalized_url)
            {
                existing.merge(entry);
            } else {
                self.entries.push(entry);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(label: &str, subtype: &str) -> EvidenceEntry {
        EvidenceEntry {
            source_id: "same".into(),
            normalized_url: "https://example.test/".into(),
            title: "title".into(),
            url: "https://example.test".into(),
            supporting_snippet: "support".into(),
            rank_decision: None,
            provider_labels: [label.into()].into(),
            source_subtypes: [subtype.into()].into(),
        }
    }
    #[test]
    fn merge_deduplicates_without_losing_provider_labels() {
        let mut provenance = TurnProvenance {
            turn_id: "t1".into(),
            entries: vec![],
        };
        let mut second = entry("two", "html");
        second.source_id = "different".into();
        provenance.merge([entry("one", "api"), second]);
        assert_eq!(provenance.entries.len(), 1);
        assert_eq!(provenance.entries[0].provider_labels.len(), 2);
        assert_eq!(provenance.entries[0].source_subtypes.len(), 2);
    }
}
