use crate::device::Device;
use crate::dispatch::{pack_rta, Program};
use crate::hw::CoreCoord;
use std::any::{Any, TypeId};
use std::collections::{btree_map::Entry, BTreeMap, HashMap};
use std::hash::Hash;
use std::io;
use std::mem::size_of;
use std::sync::{Arc, Mutex, OnceLock};

static PROGRAM_CACHE: OnceLock<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> =
    OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeArgs {
    cores: Arc<[CoreCoord]>,
    writer_bytes: usize,
    reader_bytes: usize,
    compute_bytes: usize,
    semaphores: usize,
    sem_offset: usize,
    writer_patches: Arc<[RuntimeArgPatchGroup]>,
    reader_patches: Arc<[RuntimeArgPatchGroup]>,
    compute_patches: Arc<[RuntimeArgPatchGroup]>,
    blobs: Vec<Vec<u8>>,
}

impl RuntimeArgs {
    pub(crate) fn cores(&self) -> &[CoreCoord] {
        &self.cores
    }

    pub(crate) fn blobs(&self) -> &[Vec<u8>] {
        &self.blobs
    }

    pub(crate) fn section_sizes(&self) -> (usize, usize, usize) {
        (self.writer_bytes, self.reader_bytes, self.compute_bytes)
    }

    pub(crate) fn semaphores(&self) -> usize {
        self.semaphores
    }

    pub(crate) fn sem_offset(&self) -> usize {
        self.sem_offset
    }

    #[inline]
    pub(crate) fn update_from_kernel<K>(&mut self, kernel: &impl Kernel<K>) -> io::Result<()> {
        patch_section(
            &mut self.blobs,
            &self.cores,
            &self.reader_patches,
            |core, index| kernel.reader_runtime_arg(core, index),
        )?;
        patch_section(
            &mut self.blobs,
            &self.cores,
            &self.writer_patches,
            |core, index| kernel.writer_runtime_arg(core, index),
        )?;
        patch_section(
            &mut self.blobs,
            &self.cores,
            &self.compute_patches,
            |core, index| kernel.compute_runtime_arg(core, index),
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeArgsBuilder {
    per_core: BTreeMap<CoreCoord, PerCoreRuntimeArgs>,
    semaphores: usize,
    writer_dynamic_indices: Vec<usize>,
    reader_dynamic_indices: Vec<usize>,
    compute_dynamic_indices: Vec<usize>,
}

impl RuntimeArgsBuilder {
    pub(crate) fn new(
        semaphores: usize,
        writer_dynamic_indices: Vec<usize>,
        reader_dynamic_indices: Vec<usize>,
        compute_dynamic_indices: Vec<usize>,
    ) -> Self {
        Self {
            per_core: BTreeMap::new(),
            semaphores,
            writer_dynamic_indices,
            reader_dynamic_indices,
            compute_dynamic_indices,
        }
    }

    pub(crate) fn add_core(
        &mut self,
        core: CoreCoord,
        writer: Vec<u32>,
        reader: Vec<u32>,
        compute: Vec<u32>,
    ) -> io::Result<()> {
        match self.per_core.entry(core) {
            Entry::Vacant(entry) => {
                entry.insert(PerCoreRuntimeArgs {
                    writer,
                    reader,
                    compute,
                });
                Ok(())
            }
            Entry::Occupied(entry) => Err(invalid_input(format!(
                "duplicate runtime args for core {}",
                entry.key()
            ))),
        }
    }

    pub(crate) fn build(self) -> io::Result<RuntimeArgs> {
        let Some(layout) = self.per_core.values().next() else {
            return Err(invalid_input("runtime args require at least one core"));
        };
        let semaphores = self.semaphores;
        let writer_bytes = layout.writer.len() * size_of::<u32>();
        let reader_bytes = layout.reader.len() * size_of::<u32>();
        let compute_bytes = layout.compute.len() * size_of::<u32>();
        let sem_offset = align16(writer_bytes + reader_bytes + compute_bytes);

        let writer_patches = section_patches(&self.writer_dynamic_indices, 0);
        let reader_patches = section_patches(&self.reader_dynamic_indices, writer_bytes);
        let compute_patches =
            section_patches(&self.compute_dynamic_indices, writer_bytes + reader_bytes);

        let mut cores = Vec::with_capacity(self.per_core.len());
        let mut blobs = Vec::with_capacity(self.per_core.len());

        for (core, args) in self.per_core {
            if args.writer.len() * size_of::<u32>() != writer_bytes
                || args.reader.len() * size_of::<u32>() != reader_bytes
                || args.compute.len() * size_of::<u32>() != compute_bytes
            {
                return Err(invalid_input(format!(
                    "runtime arg section lengths for core {core} do not match the first core"
                )));
            }
            cores.push(core);
            blobs.push(pack_rta(
                &args.writer,
                &args.reader,
                &args.compute,
                semaphores,
                sem_offset,
            ));
        }
        Ok(RuntimeArgs {
            cores: cores.into(),
            writer_bytes,
            reader_bytes,
            compute_bytes,
            semaphores,
            sem_offset,
            writer_patches: writer_patches.into(),
            reader_patches: reader_patches.into(),
            compute_patches: compute_patches.into(),
            blobs,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeArgPatchGroup {
    index: usize,
    offset: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PerCoreRuntimeArgs {
    writer: Vec<u32>,
    reader: Vec<u32>,
    compute: Vec<u32>,
}

fn section_patches(indices: &[usize], base_offset: usize) -> Vec<RuntimeArgPatchGroup> {
    indices
        .iter()
        .map(|&index| RuntimeArgPatchGroup {
            index,
            offset: base_offset + index * size_of::<u32>(),
        })
        .collect()
}

fn patch_section(
    blobs: &mut [Vec<u8>],
    cores: &[CoreCoord],
    groups: &[RuntimeArgPatchGroup],
    value: impl Fn(CoreCoord, usize) -> Option<u32>,
) -> io::Result<()> {
    for group in groups {
        for (core_index, blob) in blobs.iter_mut().enumerate() {
            let value = value(cores[core_index], group.index).ok_or_else(|| {
                invalid_input(format!(
                    "missing dynamic runtime arg value for index {} on core {}",
                    group.index, cores[core_index]
                ))
            })?;
            patch_u32(blob, group.offset, value)?;
        }
    }
    Ok(())
}

#[inline]
fn patch_u32(blob: &mut [u8], offset: usize, value: u32) -> io::Result<()> {
    let end = offset + size_of::<u32>();
    let blob_len = blob.len();
    let bytes = blob.get_mut(offset..end).ok_or_else(|| {
        invalid_input(format!(
            "runtime arg patch offset {offset} exceeds blob size {blob_len}"
        ))
    })?;
    bytes.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn align16(value: usize) -> usize {
    (value + 15) & !15
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

pub(crate) trait Kernel<K = ()> {
    fn program_key(&self) -> K;

    fn build_program(&self) -> io::Result<Program> {
        Err(invalid_input("kernel does not define a cached program"))
    }

    fn run(&self, device: &mut Device) -> io::Result<()>
    where
        Self: Sized + 'static,
        K: Eq + Hash + Send + Sync + 'static,
    {
        let key = self.program_key();
        let caches = PROGRAM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut caches = caches.lock().map_err(|_| {
            io::Error::other(format!(
                "{} cache is poisoned",
                std::any::type_name::<Self>()
            ))
        })?;
        let cache = caches
            .entry(TypeId::of::<Self>())
            .or_insert_with(|| Box::new(HashMap::<K, Arc<Program>>::new()));
        let cache = cache
            .downcast_mut::<HashMap<K, Arc<Program>>>()
            .ok_or_else(|| {
                io::Error::other(format!(
                    "{} cache key type changed",
                    std::any::type_name::<Self>()
                ))
            })?;

        let program = if let Some(program) = cache.get(&key).map(Arc::clone) {
            program
        } else {
            let program = Arc::new(self.build_program()?);
            cache.insert(key, Arc::clone(&program));
            program
        };
        device.run_cached_program(program, |runtime_args| {
            runtime_args.update_from_kernel(self)
        })
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, _index: usize) -> Option<u32> {
        None
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, _index: usize) -> Option<u32> {
        None
    }

    #[inline]
    fn compute_runtime_arg(&self, _core: CoreCoord, _index: usize) -> Option<u32> {
        None
    }
}

pub(crate) fn select_worker_cores(
    available: &[CoreCoord],
    tile_count: usize,
) -> io::Result<Vec<CoreCoord>> {
    if available.is_empty() {
        return Err(invalid_input("no worker cores are available"));
    }
    let n_cores = available.len().min(tile_count.max(1));
    Ok(available[..n_cores].to_vec())
}

pub(crate) fn split_tile_range(
    tile_count: u32,
    core_index: usize,
    n_cores: usize,
) -> io::Result<(u32, u32)> {
    if n_cores == 0 {
        return Err(invalid_input("tile range requires at least one core"));
    }
    if core_index >= n_cores {
        return Err(invalid_input(format!(
            "core index {core_index} is out of range for {n_cores} cores"
        )));
    }

    let tile_count = usize::try_from(tile_count)
        .map_err(|_| invalid_input(format!("tile count does not fit in usize: {tile_count}")))?;
    let base = tile_count / n_cores;
    let remainder = tile_count % n_cores;
    let count = base + usize::from(core_index < remainder);
    let offset = core_index
        .checked_mul(base)
        .and_then(|value| value.checked_add(core_index.min(remainder)))
        .ok_or_else(|| invalid_input("tile range offset overflow"))?;
    Ok((
        u32::try_from(offset)
            .map_err(|_| invalid_input(format!("tile offset does not fit in u32: {offset}")))?,
        u32::try_from(count)
            .map_err(|_| invalid_input(format!("tile count does not fit in u32: {count}")))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestKernel;

    impl Kernel for TestKernel {
        fn program_key(&self) {}

        fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
            (index == 0).then_some(0x2222)
        }

        fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
            (index == 1).then_some(0x1111)
        }
    }

    #[test]
    fn runtime_args_update_patches_blobs_by_section_and_index() {
        let mut builder = RuntimeArgsBuilder::new(0, vec![1], vec![0], Vec::new());
        builder
            .add_core(CoreCoord { x: 1, y: 2 }, vec![7, 0], vec![0, 9], Vec::new())
            .expect("add core");

        let mut runtime_args = builder.build().expect("lower");
        runtime_args
            .update_from_kernel(&TestKernel)
            .expect("update");

        assert_eq!(&runtime_args.blobs()[0][4..8], &0x1111u32.to_le_bytes());
        assert_eq!(&runtime_args.blobs()[0][8..12], &0x2222u32.to_le_bytes());
    }

    #[test]
    fn split_tile_range_balances_remainder_across_early_cores() {
        assert_eq!(split_tile_range(10, 0, 3).expect("range"), (0, 4));
        assert_eq!(split_tile_range(10, 1, 3).expect("range"), (4, 3));
        assert_eq!(split_tile_range(10, 2, 3).expect("range"), (7, 3));
    }
}
