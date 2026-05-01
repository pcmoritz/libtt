use crate::dispatch::{pack_rta, Program};
use crate::hw::CoreCoord;
use std::collections::BTreeMap;
use std::io;
use std::mem::size_of;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeArgs {
    cores: Arc<[CoreCoord]>,
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

    #[inline]
    pub(crate) fn update_from_kernel(&mut self, kernel: &impl Kernel) -> io::Result<()> {
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
}

impl RuntimeArgsBuilder {
    pub(crate) fn new(semaphores: usize) -> Self {
        Self {
            per_core: BTreeMap::new(),
            semaphores,
        }
    }

    pub(crate) fn add_core(
        &mut self,
        core: CoreCoord,
        writer: Vec<Option<u32>>,
        reader: Vec<Option<u32>>,
        compute: Vec<Option<u32>>,
    ) -> io::Result<()> {
        if self.per_core.contains_key(&core) {
            return Err(invalid_input(format!(
                "duplicate runtime args for core {core}"
            )));
        }

        self.per_core.insert(
            core,
            PerCoreRuntimeArgs {
                writer: RuntimeArgSection::from_args(writer),
                reader: RuntimeArgSection::from_args(reader),
                compute: RuntimeArgSection::from_args(compute),
            },
        );
        Ok(())
    }

    pub(crate) fn build(self) -> io::Result<RuntimeArgs> {
        Ok(self.lower()?.0)
    }

    pub(crate) fn apply_to_program(self, program: &mut Program) -> io::Result<()> {
        let semaphores = self.semaphores;
        let (runtime_args, writer_args, reader_args, compute_args) = self.lower()?;
        program.writer_args = writer_args;
        program.reader_args = reader_args;
        program.compute_args = compute_args;
        program.semaphores = semaphores;
        program.runtime_args = Some(runtime_args);
        Ok(())
    }

    fn lower(self) -> io::Result<(RuntimeArgs, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let writer_args = self
            .per_core
            .values()
            .next()
            .map(|args| args.writer.values.clone())
            .unwrap_or_default();
        let reader_args = self
            .per_core
            .values()
            .next()
            .map(|args| args.reader.values.clone())
            .unwrap_or_default();
        let compute_args = self
            .per_core
            .values()
            .next()
            .map(|args| args.compute.values.clone())
            .unwrap_or_default();

        let mut cores = Vec::with_capacity(self.per_core.len());
        let mut blobs = Vec::with_capacity(self.per_core.len());
        let mut writer_patches = Vec::with_capacity(self.per_core.len());
        let mut reader_patches = Vec::with_capacity(self.per_core.len());
        let mut compute_patches = Vec::with_capacity(self.per_core.len());

        for (core, args) in self.per_core {
            let writer_bytes = args.writer.len() * size_of::<u32>();
            let reader_bytes = args.reader.len() * size_of::<u32>();
            let compute_bytes = args.compute.len() * size_of::<u32>();
            let sem_off = align16(writer_bytes + reader_bytes + compute_bytes);

            writer_patches.push(section_patches(&args.writer, 0));
            reader_patches.push(section_patches(&args.reader, writer_bytes));
            compute_patches.push(section_patches(&args.compute, writer_bytes + reader_bytes));
            cores.push(core);
            blobs.push(pack_rta(
                &args.writer.values,
                &args.reader.values,
                &args.compute.values,
                self.semaphores,
                sem_off,
            ));
        }

        let runtime_args = RuntimeArgs {
            cores: cores.into(),
            writer_patches: patch_groups(writer_patches).into(),
            reader_patches: patch_groups(reader_patches).into(),
            compute_patches: patch_groups(compute_patches).into(),
            blobs,
        };

        Ok((runtime_args, writer_args, reader_args, compute_args))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeArgPatchGroup {
    index: usize,
    offsets_by_core: Vec<Option<usize>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PerCoreRuntimeArgs {
    writer: RuntimeArgSection,
    reader: RuntimeArgSection,
    compute: RuntimeArgSection,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RuntimeArgSection {
    values: Vec<u32>,
    dynamic_indices: Vec<usize>,
}

impl RuntimeArgSection {
    fn from_args(args: Vec<Option<u32>>) -> Self {
        let mut section = Self::default();
        for arg in args {
            match arg {
                Some(value) => section.push(value),
                None => section.push_dynamic(),
            }
        }
        section
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    // Static runtime args are fixed when the cached program is built.
    fn push(&mut self, value: u32) {
        self.values.push(value);
    }

    // Dynamic runtime args reserve a slot that is patched per launch.
    fn push_dynamic(&mut self) {
        let index = self.values.len();
        self.values.push(0);
        self.dynamic_indices.push(index);
    }
}

fn section_patches(args: &RuntimeArgSection, base_offset: usize) -> Vec<(usize, usize)> {
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
            groups
                .entry(index)
                .or_insert_with(|| vec![None; core_count])[core_index] = Some(offset);
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
    value: impl Fn(CoreCoord, usize) -> Option<u32>,
) -> io::Result<()> {
    for group in groups {
        for (core_index, offset) in group.offsets_by_core.iter().enumerate() {
            let Some(offset) = offset else {
                continue;
            };
            let value = value(cores[core_index], group.index).ok_or_else(|| {
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

pub(crate) trait Kernel {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct TestKernel;

    impl Kernel for TestKernel {
        fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
            (index == 0).then_some(0x2222)
        }

        fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
            (index == 1).then_some(0x1111)
        }
    }

    #[test]
    fn runtime_args_update_patches_blobs_by_section_and_index() {
        let mut builder = RuntimeArgsBuilder::new(0);
        builder
            .add_core(
                CoreCoord { x: 1, y: 2 },
                vec![Some(7), None],
                vec![None, Some(9)],
                Vec::new(),
            )
            .expect("add core");

        let mut runtime_args = builder.build().expect("lower");
        runtime_args
            .update_from_kernel(&TestKernel)
            .expect("update");

        assert_eq!(&runtime_args.blobs()[0][4..8], &0x1111u32.to_le_bytes());
        assert_eq!(&runtime_args.blobs()[0][8..12], &0x2222u32.to_le_bytes());
    }
}
