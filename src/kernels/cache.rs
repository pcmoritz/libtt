use crate::dispatch::{pack_rta, Program, RuntimeArgs};
use crate::hw::CoreCoord;
use crate::kernels::kernel::Kernel;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::io;
use std::mem::size_of;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RuntimeArgList {
    values: Vec<u32>,
    dynamic_indices: Vec<usize>,
}

impl RuntimeArgList {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push(&mut self, value: u32) {
        self.values.push(value);
    }

    pub(crate) fn push_dynamic(&mut self) {
        let index = self.values.len();
        self.values.push(0);
        self.dynamic_indices.push(index);
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn values(&self) -> &[u32] {
        &self.values
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PerCoreRuntimeArgs {
    pub(crate) core: CoreCoord,
    pub(crate) writer: RuntimeArgList,
    pub(crate) reader: RuntimeArgList,
    pub(crate) compute: RuntimeArgList,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeArgPatchGroup {
    pub(crate) index: usize,
    pub(crate) offsets_by_core: Vec<Option<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedRuntimeArgs {
    pub(crate) runtime_args: RuntimeArgs,
    pub(crate) writer_args: Vec<u32>,
    pub(crate) reader_args: Vec<u32>,
    pub(crate) compute_args: Vec<u32>,
}

pub(crate) struct ProgramCache<K> {
    name: &'static str,
    entries: OnceLock<Mutex<HashMap<K, Program>>>,
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
    ) -> io::Result<Program> {
        let entries = self.entries.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(program) = entries
            .lock()
            .map_err(|_| io::Error::other(format!("{} cache is poisoned", self.name)))?
            .get(&key)
            .cloned()
        {
            return Ok(program);
        }

        let program = build()?;
        entries
            .lock()
            .map_err(|_| io::Error::other(format!("{} cache is poisoned", self.name)))?
            .insert(key, program.clone());
        Ok(program)
    }
}

impl RuntimeArgs {
    pub(crate) fn from_per_core(
        mut per_core: Vec<PerCoreRuntimeArgs>,
        semaphores: usize,
    ) -> io::Result<PackedRuntimeArgs> {
        per_core.sort_unstable_by_key(|args| args.core);

        let writer_args = per_core
            .first()
            .map(|args| args.writer.values().to_vec())
            .unwrap_or_default();
        let reader_args = per_core
            .first()
            .map(|args| args.reader.values().to_vec())
            .unwrap_or_default();
        let compute_args = per_core
            .first()
            .map(|args| args.compute.values().to_vec())
            .unwrap_or_default();

        let mut cores = Vec::with_capacity(per_core.len());
        let mut blobs = Vec::with_capacity(per_core.len());
        let mut writer_patches = Vec::with_capacity(per_core.len());
        let mut reader_patches = Vec::with_capacity(per_core.len());
        let mut compute_patches = Vec::with_capacity(per_core.len());

        for args in per_core {
            let writer_bytes = args.writer.len() * size_of::<u32>();
            let reader_bytes = args.reader.len() * size_of::<u32>();
            let compute_bytes = args.compute.len() * size_of::<u32>();
            let sem_off = align16(writer_bytes + reader_bytes + compute_bytes);

            writer_patches.push(section_patches(&args.writer, 0));
            reader_patches.push(section_patches(&args.reader, writer_bytes));
            compute_patches.push(section_patches(
                &args.compute,
                writer_bytes + reader_bytes,
            ));
            cores.push(args.core);
            blobs.push(pack_rta(
                args.writer.values(),
                args.reader.values(),
                args.compute.values(),
                semaphores,
                sem_off,
            ));
        }

        let runtime_args = RuntimeArgs {
            cores,
            blobs,
            writer_patches: patch_groups(writer_patches),
            reader_patches: patch_groups(reader_patches),
            compute_patches: patch_groups(compute_patches),
        };

        Ok(PackedRuntimeArgs {
            runtime_args,
            writer_args,
            reader_args,
            compute_args,
        })
    }

    pub(crate) fn update_from_kernel(&self, kernel: &impl Kernel) -> io::Result<Self> {
        let mut next = self.clone();
        patch_section(
            &mut next.blobs,
            &self.cores,
            &self.reader_patches,
            |core, index| kernel.reader_runtime_arg(core, index),
        )?;
        patch_section(
            &mut next.blobs,
            &self.cores,
            &self.writer_patches,
            |core, index| kernel.writer_runtime_arg(core, index),
        )?;
        patch_section(
            &mut next.blobs,
            &self.cores,
            &self.compute_patches,
            |core, index| kernel.compute_runtime_arg(core, index),
        )?;
        Ok(next)
    }
}

impl Program {
    pub(crate) fn update_runtime_args_from_kernel(
        &self,
        kernel: &impl Kernel,
    ) -> io::Result<Self> {
        let mut program = self.clone();
        if let Some(runtime_args) = &self.runtime_args {
            program.runtime_args = Some(runtime_args.update_from_kernel(kernel)?);
        }
        Ok(program)
    }
}

fn section_patches(args: &RuntimeArgList, base_offset: usize) -> Vec<(usize, usize)> {
    args.dynamic_indices
        .iter()
        .map(|&index| (index, base_offset + index * size_of::<u32>()))
        .collect()
}

fn patch_groups(per_core: Vec<Vec<(usize, usize)>>) -> Vec<RuntimeArgPatchGroup> {
    let core_count = per_core.len();
    let mut groups = BTreeMap::<usize, Vec<Option<usize>>>::new();
    for (core_index, patches) in per_core.into_iter().enumerate() {
        for (index, offset) in patches {
            groups.entry(index).or_insert_with(|| vec![None; core_count])[core_index] =
                Some(offset);
        }
    }
    groups
        .into_iter()
        .map(|(index, offsets_by_core)| RuntimeArgPatchGroup {
            index,
            offsets_by_core,
        })
        .collect()
}

fn patch_section(
    blobs: &mut [Vec<u8>],
    cores: &[CoreCoord],
    groups: &[RuntimeArgPatchGroup],
    value: impl Fn(CoreCoord, usize) -> io::Result<Option<u32>>,
) -> io::Result<()> {
    for group in groups {
        for (core_index, offset) in group.offsets_by_core.iter().enumerate() {
            let Some(offset) = offset else {
                continue;
            };
            let value = value(cores[core_index], group.index)?.ok_or_else(|| {
                invalid_input(format!(
                    "missing dynamic runtime arg value for index {} on core {}",
                    group.index, cores[core_index]
                ))
            })?;
            patch_u32(&mut blobs[core_index], *offset, value)?;
        }
    }
    Ok(())
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    struct TestKernel;

    impl Kernel for TestKernel {
        fn program(&self, _device: &Device) -> io::Result<Program> {
            Ok(Program::default())
        }

        fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> io::Result<Option<u32>> {
            Ok((index == 0).then_some(0x2222))
        }

        fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> io::Result<Option<u32>> {
            Ok((index == 1).then_some(0x1111))
        }
    }

    #[test]
    fn runtime_args_update_patches_blobs_by_section_and_index() {
        let mut writer = RuntimeArgList::new();
        writer.push(7);
        writer.push_dynamic();
        let mut reader = RuntimeArgList::new();
        reader.push_dynamic();
        reader.push(9);

        let packed = RuntimeArgs::from_per_core(
            vec![PerCoreRuntimeArgs {
                core: CoreCoord { x: 1, y: 2 },
                writer,
                reader,
                compute: RuntimeArgList::new(),
            }],
            0,
        )
        .expect("lower");

        let updated = packed
            .runtime_args
            .update_from_kernel(&TestKernel)
            .expect("update");

        assert_eq!(&updated.blobs[0][4..8], &0x1111u32.to_le_bytes());
        assert_eq!(&updated.blobs[0][8..12], &0x2222u32.to_le_bytes());
    }
}
