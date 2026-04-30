use crate::dispatch::Program;
use std::collections::HashMap;
use std::hash::Hash;
use std::io;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) struct ProgramCache<K> {
    name: &'static str,
    entries: OnceLock<Mutex<HashMap<K, Arc<Program>>>>,
}

impl<K> ProgramCache<K>
where
    K: Eq + Hash + Clone,
{
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            entries: OnceLock::new(),
        }
    }

    pub(crate) fn get_or_insert_with(
        &self,
        key: K,
        build: impl FnOnce() -> io::Result<Program>,
    ) -> io::Result<Arc<Program>> {
        let entries = self.entries.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(program) = entries
            .lock()
            .map_err(|_| io::Error::other(format!("{} cache is poisoned", self.name)))?
            .get(&key)
            .map(Arc::clone)
        {
            return Ok(program);
        }

        let program = Arc::new(build()?);
        entries
            .lock()
            .map_err(|_| io::Error::other(format!("{} cache is poisoned", self.name)))?
            .insert(key, Arc::clone(&program));
        Ok(program)
    }
}
