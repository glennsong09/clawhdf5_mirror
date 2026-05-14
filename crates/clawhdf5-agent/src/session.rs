//! Session tracking cache and data structures.

/// A single session entry.
#[derive(Debug, Clone)]
pub struct SessionEntry {
    pub id: String,
    pub start_idx: u64,
    pub end_idx: u64,
    pub channel: String,
    pub ts: f64,
}

/// In-memory cache for the /sessions group.
#[derive(Debug, Clone)]
pub struct SessionCache {
    pub entries: Vec<SessionEntry>,
    pub summaries: Vec<String>,
}

impl SessionCache {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            summaries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a new session with its summary.
    pub fn add(
        &mut self,
        id: &str,
        start_idx: usize,
        end_idx: usize,
        channel: &str,
        summary: &str,
    ) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            * 1_000_000.0; // microseconds
        self.entries.push(SessionEntry {
            id: id.to_string(),
            start_idx: start_idx as u64,
            end_idx: end_idx as u64,
            channel: channel.to_string(),
            ts,
        });
        self.summaries.push(summary.to_string());
    }

    /// Return the ID of the most recently added session, if any.
    pub fn latest_session_id(&self) -> Option<&str> {
        self.entries.last().map(|e| e.id.as_str())
    }

    /// Find the summary for a session by ID.
    pub fn find_summary(&self, session_id: &str) -> Option<&str> {
        self.entries
            .iter()
            .position(|e| e.id == session_id)
            .map(|idx| self.summaries[idx].as_str())
    }
}

impl Default for SessionCache {
    fn default() -> Self {
        Self::new()
    }
}
